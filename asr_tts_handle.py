from gtts import gTTS
from PIL import Image
import tempfile
import speech_recognition as sr
from voice_utils import register_voice, verify_voice
from scene_description import generate_scene_description
from read_text import read_text


def handle_input(audio_path: str, image: Image.Image) -> tuple[str, str | None]:
    """
    Processes audio command and image input, now using Google Speech Recognition.
    Returns: A tuple (text_response, audio_file_path_or_none)
    """
    transcribed_text = ""
    audio_file_path = None

    # 1. SECURITY STEP: Verify the Speaker
    is_verified, verification_status_message = verify_voice(audio_path)

    if not is_verified:
        # If verification fails, stop and return the failure message
        return verification_status_message, None

    # 2. Handle Audio Transcription using speech_recognition (only if verified)
    if audio_path is None:
        transcribed_text = "No audio command received."
    else:
        r = sr.Recognizer()
        try:
            # Use sr.AudioFile to read the audio path provided by Gradio
            with sr.AudioFile(audio_path) as source:
                audio = r.record(source)
                # Use Google Web Speech API for transcription
                transcribed_text = r.recognize_google(audio).strip().lower()
        except sr.UnknownValueError:
            transcribed_text = "Speech Recognition could not understand audio."
        except sr.RequestError as e:
            transcribed_text = f"Could not request results from Google Speech Recognition service; {e}"
        except Exception as e:
            transcribed_text = f"General error during speech transcription: {e}"


    # 3. Handle Multimodal Command Logic
    output_message = f"{verification_status_message}\n\nCommand received: '{transcribed_text}'"
    image_status = "Image successfully captured." if image is not None else "Warning: No image captured."

    #scene description/reading text using the model
    try:
        if 'read' in transcribed_text.lower():
            friday_response = read_text(image)
        else:
            friday_response = generate_scene_description(transcribed_text, image)

    except Exception as e:
        friday_response = "Error: Not able to describe the screen"

    output_message += f"\n{image_status}\n Friday: {friday_response}"

    # text-to-speech
    narration_text = f"The analysis of the captured scene is: {friday_response}"
    try:
        output_message += "inside try block"
        tts = gTTS(text=narration_text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        audio_file_path = temp_file.name
        temp_file.close()
        output_message += "after close()"
        tts.save(audio_file_path)
        output_message += f"after tts save: {audio_file_path}"
    except Exception as tts_e:
        output_message += f"\n\n🚨 TTS Error: Could not generate audio. Details: {tts_e}"
        audio_file_path = None

    return output_message, audio_file_path

