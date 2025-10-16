from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
from PIL import Image

# --- Loading the model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🔹 Using device: {device}")

model_path = "./llava_local"
processor = LlavaProcessor.from_pretrained(model_path, use_fast=True)
model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16).to(device).eval()


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
