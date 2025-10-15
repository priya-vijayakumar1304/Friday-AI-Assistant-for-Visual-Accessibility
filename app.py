from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image
import gradio as gr
import os
import io
from PIL import Image
from gtts import gTTS
import tempfile
import librosa
import numpy as np
import speech_recognition as sr
import easyocr
import numpy as np


# --- Loading the model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

model_path = "./llava_local"
processor = LlavaProcessor.from_pretrained(model_path, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16).to(device).eval()


# --- Speaker Verification Configuration ---

# File path to save the voice signature (MFCC mean)
VOICE_SIGNATURE_PATH = "voice_signature.npy"
# Threshold for Euclidean distance (lower is closer match)
VERIFICATION_THRESHOLD = 60.0



# --- 1. Voice Signature Functions ---

def extract_mfccs(audio_path: str) -> np.ndarray | None:
    """Extracts mean MFCC features from an audio file."""
    try:
        # Load audio file (mono, 16kHz)
        y, sr = librosa.load(audio_path, sr=16000)
        # Extract MFCCs (13 coefficients by default)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Return the mean of all MFCCs across time (the 'signature')
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error extracting MFCCs: {e}")
        return None


def register_voice(audio_path: str | None) -> str:
    """Registers the speaker by saving their voice signature."""
    if audio_path is None:
        return "Registration failed: Please record a sample of your voice [5-10sec]."

    signature = extract_mfccs(audio_path)

    if signature is not None:
        try:
            np.save(VOICE_SIGNATURE_PATH, signature)
            return f"Voice registered successfully! Signature saved to '{VOICE_SIGNATURE_PATH}'. You can now use the Friday AI Assistant."
        except Exception as e:
            return f"Registration failed: Could not save signature. Details: {e}"
    else:
        return "Registration failed: Could not process audio features."


def verify_voice(audio_path: str | None) -> tuple[bool, str]:
    """Verifies the speaker against the stored signature."""
    if not os.path.exists(VOICE_SIGNATURE_PATH):
        return False, "Verification required: No voice signature found. Please register your voice first."

    if audio_path is None:
        return False, "Verification failed: No audio provided for verification."

    stored_signature = np.load(VOICE_SIGNATURE_PATH)
    current_signature = extract_mfccs(audio_path)

    if current_signature is None:
        return False, "Verification failed: Could not process input audio features."

    # Calculate Euclidean distance between the stored and current signatures
    distance = np.linalg.norm(stored_signature - current_signature)

    # Simple verification logic: Check if distance is below the threshold
    if distance < VERIFICATION_THRESHOLD:
        return True, f"Voice Verification Successful!" # Distance: {distance:.2f}, Threshold: {VERIFICATION_THRESHOLD}
    else:
        return False, f"Verification failed: Speaker mismatch detected." # Distance: {distance:.2f} (Threshold: {VERIFICATION_THRESHOLD}



# --- 2. Main Command Function ---

def generate_scene_description(command: str, image: Image.Image) -> str:
    """
    Generates a scene description based on a command and an image.
    """
    user_prompt = "answer very precisely and concisely, and in less than 70 words for the command: " + command
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image"},
              ],
        },
    ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)

    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    scene_descripton = processor.decode(output[0][2:], skip_special_tokens=True)[len(user_prompt)+13:]

    return scene_descripton


def read_text(image: Image.Image) -> str :
    gpu = True if device == "cuda" else False

    reader = easyocr.Reader(['en'], gpu=gpu)

    CONF_THRESHOLD = 0.40   # Ignore text below this confidence
    Y_THRESHOLD = 15        # For grouping text in the same line

    image_np = np.array(image)
    # Run OCR
    results = reader.readtext(image_np)

    # --- Filter low-confidence results ---
    results = [r for r in results if r[2] >= CONF_THRESHOLD]

    if not results:
        return "no text found"

    # --- Group results into lines based on y-coordinate ---
    def group_lines(results, y_threshold=Y_THRESHOLD):
        # Sort all results top-to-bottom
        results = sorted(results, key=lambda x: x[0][0][1])

        lines = []
        current_line = []
        last_y = None

        for (bbox, text, conf) in results:
            y = bbox[0][1]
            if last_y is None or abs(y - last_y) < y_threshold:
                current_line.append((bbox, text, conf))
            else:
                lines.append(current_line)
                current_line = [(bbox, text, conf)]
            last_y = y

        if current_line:
            lines.append(current_line)
        return lines

    # --- Reconstruct text line-by-line ---
    lines = group_lines(results)

    final_text = []
    for line in lines:
        # Sort left to right within a line
        line = sorted(line, key=lambda x: x[0][0][0])
        joined = " ".join([w[1] for w in line])
        final_text.append(joined)

    return "\n".join(final_text) 


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

    #scene description using the model
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

# --- 3. Gradio Blocks Interface ---

# --- Custom CSS for Styling ---
CUSTOM_CSS = """
/* 1. Global Styles and Font */
.gradio-container {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background-color: #f7f9fb; /* Lighter background */
    padding: 20px;
}
.gradio-tabs {
    border-radius: 20px; /* Larger radius */
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15); /* Stronger, softer shadow */
    background-color: white;
    padding: 30px; /* More internal padding */
    border: none;
}
h1 {
    color: #0d47a1; /* Deep blue title */
    text-align: center;
    font-weight: 700;
    margin-bottom: 20px;
}
h2, h3 {
    color: #374151; /* Subdued text color */
    border-bottom: 1px solid #e5e7eb; /* Thinner, lighter separator */
    padding-bottom: 8px;
    margin-top: 15px;
}

/* 2. Tab Styling */
.gradio-tabs button {
    font-weight: 700;
    border-radius: 10px 10px 0 0 !important;
    color: #6b7280; /* Gray inactive tab text */
    transition: all 0.2s ease-in-out;
}
.gradio-tabs button.selected {
    color: #4c51bf !important; /* Indigo active tab text */
    background-color: #f0f4f8 !important; /* Light background for active tab */
    border-bottom: 3px solid #4c51bf !important; /* Highlight bottom border */
}

/* 3. Button Styling (Enhanced) */
.gr-button {
    background-color: #4c51bf; /* Primary indigo color */
    color: white;
    border: none;
    border-radius: 12px; /* Slightly larger radius for button */
    padding: 14px 25px; /* More padding */
    font-weight: 700;
    letter-spacing: 0.5px;
    transition: all 0.3s ease;
    text-transform: uppercase;
}
.gr-button:hover {
    background-color: #3c40a5; /* Darker indigo on hover */
    transform: translateY(-2px); /* Slight lift effect */
    box-shadow: 0 8px 20px rgba(76, 81, 191, 0.5); /* Stronger shadow on hover */
}

/* 4. Input/Output Elements */
.gr-textbox textarea, .gr-audio, .gr-image {
    border-radius: 12px;
    border: 1px solid #d1d5db; /* Lighter, cleaner border color */
    background-color: #ffffff;
    padding: 15px;
    transition: border-color 0.3s;
}
.gr-textbox textarea:focus, .gr-audio:focus, .gr-image:focus {
    border-color: #4c51bf; /* Highlight border on focus */
    box-shadow: 0 0 0 2px rgba(76, 81, 191, 0.2);
    outline: none;
}
.gr-audio-player audio {
    border-radius: 12px;
}
"""


with gr.Blocks(title="Friday, A Secure Multimodal Voice-Vision Assistant", css=CUSTOM_CSS) as demo:
    gr.Markdown(f"# Friday, A Secure Multimodal AI Assistant\n")

    with gr.Tab("💬 Friday, your AI Assistant"):
        gr.Markdown("## Scene description\n **If you are using Friday the first time, please register your voice for authentication and security purpose**")
        gr.Markdown("\nRecord a voice command (e.g., 'Friday, Describe the scene') and capture an image simultaneously. The assistant will transcribe the command, analyze the image, and narrate the description. **Only commands from the registered voice will be processed.**")

        with gr.Row(equal_height=True):
            # Column 1: Inputs - Voice & Camera
            with gr.Column(scale=1):
                command_audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Voice Command")
                image_input = gr.Image(sources=["webcam"], type="pil", label="📸 Live Image Capture", mirror_webcam=False)
                run_button = gr.Button("▶️ Submit")
            # Column 2: Outputs - Text & Narration
            with gr.Column(scale=1):
                output_message_box = gr.Textbox(lines=10, label="Assistant Response")
                audio_output = gr.Audio(label="Scene Narration", type="filepath", autoplay=True)
        run_button.click(
            fn=handle_input,
            inputs=[command_audio_input, image_input],
            outputs=[output_message_box, audio_output]
        )

    with gr.Tab("🎤 Voice Registration"):
        gr.Markdown("## Register Your Voice\nRecord a short (5-10 second) sample of your voice to create your security signature.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                register_audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record Registration Sample")
                register_button = gr.Button("🔒 Save My Voice Signature")
            with gr.Column(scale=1):
                registration_status = gr.Textbox(label="Registration Status", lines=2, interactive=False)

        register_button.click(
            fn=register_voice,
            inputs=[register_audio_input],
            outputs=[registration_status]
        )

        gr.Markdown(f"Note: Your current voice signature is used to verify your identity when using the Friday AI Assistant.")

demo.launch(share=True)


