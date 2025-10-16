import gradio as gr
from PIL import Image
import os
from asr_tts_handle import handle_input
from voice_utils import register_voice

def load_css_from_file(filepath):
    with open(filepath, 'r') as file:
        return file.read()

custom_css = load_css_from_file("custom.css")


with gr.Blocks(title="Friday, A Secure Multimodal Voice-Vision Assistant", css=custom_css) as demo:
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