# Friday - A Secure Multimodal AI Assistant for Visual Accessibility

## Description:
This application assists visually impaired users by describing their surroundings through voice. The user can use simple voice commands such as “What’s around me?” or “Describe this scene.” an image captured using the device camera, processes it with a vision-language model, and provides a spoken description of what’s detected — people, objects, actions, and scene context.

It aims to create independence by enabling users to understand their environment, powered by AI-driven image captioning and text-to-speech (TTS) technologies.
[This project demonstrates **loading, saving and deploying** the `llava-hf/llava-interleave-qwen-0.5b-hf` model locally for **low-latency inference** using **Gradio**.]

## Demo

[![Video Title](https://img.youtube.com/vi/4nwuTbgo8pY/0.jpg)](https://www.youtube.com/watch?v=4nwuTbgo8pY)

## Project structure:
```
├── load_model_and_save.ipynb # Loads and saves the LLaVA model locally
├── app.ipynb # Deploys the model using a Gradio interface
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
- **Dependencies:** Listed in `requirements.txt`

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
- Launch a Gradio interface for text and image interaction
You’ll get a public Gradio link to interact with Friday directly from your browser.


