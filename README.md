# Friday - A Secure Multimodal AI Assistant for Visual Accessibility

## Description:
This application assists visually impaired users by describing their surroundings through voice. The user can use simple voice commands such as “What’s around me?” or “Describe this scene.” and an image captured using the device camera, processes it with a vision-language model, and provides a spoken description of what’s detected — people, objects, actions, and scene context.

It aims to create independence by enabling users to understand their environment, powered by AI-driven image captioning and text-to-speech (TTS) technologies.

This project demonstrates **loading, saving and deploying** the `llava-hf/llava-interleave-qwen-0.5b-hf` model locally for **low-latency inference** using **Gradio**. It integrates **image understanding**, **text processing**, and **voice recognition with authentication**, allowing users to interact naturally and securely.

## Demo

[![Video Title](https://img.youtube.com/vi/4nwuTbgo8pY/0.jpg)](https://www.youtube.com/watch?v=4nwuTbgo8pY)

## Project structure:
```
├── load_model_and_save.ipynb # Loads and saves the LLaVA model locally
├── app.ipynb # Loads the saved model, adds voice features & Gradio frontend
├── requirements.txt # List of dependencies
└── README.md # Project documentation
```
## Tech Stack

- **Model:** [LLaVA Interleave Qwen 0.5B](https://huggingface.co/llava-hf/llava-interleave-qwen-0.5b-hf)  
- **Environment:** Google Colab (T4 GPU runtime with CUDA)  
- **Frameworks:**  
  - [Transformers (Hugging Face)](https://github.com/huggingface/transformers)  
  - [PyTorch](https://pytorch.org/)  
  - [Gradio](https://gradio.app/)
  - [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for voice commands  
  - [gTTS](https://pypi.org/project/gTTS/) or [pyttsx3](https://pypi.org/project/pyttsx3/) for text-to-speech feedback   
- **Dependencies:** Listed in `requirements.txt`

## Key Features

- **Multimodal Interaction:** Combine **audio + image** input for smarter responses.  
- **Voice Command Support:** Control Friday with spoken commands.  
- **Voice Registration & Verification:** Securely register your voice and authenticate before running actions.  
- **Local Model Loading:** Faster inference by loading saved models from disk.  
- **Low Latency Deployment:** Uses locally saved model and GPU acceleration for real-time responses.  

---

##  Setup & Usage Instructions
Change the runtime to T4 GPU in google colab

### 1. Clone this Repository
```bash
git clone https://github.com/<your-username>/Friday-AI-Assistant.git
cd Friday-AI-Assistant
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Load and Save the Model
Open `load_model_and_save.ipynb` in Google Colab and run all cells.
This notebook will:
Download the pretrained `llava-hf/llava-interleave-qwen-0.5b-hf` model
Save it locally in a specified directory (for faster access)

### 4. Run the App
Open `app.ipynb` in Colab and run the cells to:
- Load the locally saved model
- Launch a Gradio interface for audio and image interaction
You’ll get a public Gradio link to interact with Friday directly from your browser.

### Voice Registration & Verification
Friday includes a simple voice authentication system:
- Register your voice – Speak a chosen phrase to create a unique voice profile.
- Verify on command – Friday compares your live voice to the registered one before executing actions.
- Secure AI Access – Prevents unauthorized voice access while keeping interactions natural.
- Voice data is processed locally — not sent to external APIs for privacy.

## Future Enhancements
Planned features for the next version of Friday:
**Live Video Stream Processing:** Capture video or webcam feed in real time and provide live narration or scene understanding.
**Object & Text Detection:** Integrate OCR and object detection for richer visual context.
**Continuous Listening Mode:** Enable wake-word detection (e.g., “Hey Friday”) for a hands-free experience.
**Offline Mode / Edge Deployment:** Optimize the model to run efficiently on edge devices for offline accessibility.
**Multilingual Support:** Expand voice and text capabilities to multiple languages.

## Acknowledgments

- **[Hugging Face](https://huggingface.co)** - LLaVA model  
- **[Gradio](https://gradio.app)** - user interface  
- **[Google Colab](https://colab.research.google.com)** - GPU runtime support  
- **[SpeechRecognition](https://pypi.org/project/SpeechRecognition/)** - voice handling
- **[gTTS](https://pypi.org/project/gTTS/)** — text-to-speech functionality

 ## License

This project is licensed under the [**MIT License**](https://opensource.org/licenses/MIT) - feel free to use and modify it for your own research or applications.

---

