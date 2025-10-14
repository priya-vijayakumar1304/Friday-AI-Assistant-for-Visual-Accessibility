from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import os


MODEL_NAME = "llava-hf/llava-interleave-qwen-0.5b-hf"
LOCAL_PATH = "./llava_local"

print(f"Downloading LLaVA model from Hugging Face: {MODEL_NAME}")
processor = LlavaProcessor.from_pretrained(MODEL_NAME)
model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        dtype=torch.float16
)

print("Saving model and processor locally...")
os.makedirs(LOCAL_PATH, exist_ok=True)
processor.save_pretrained(LOCAL_PATH)
model.save_pretrained(LOCAL_PATH)
print(f"Model saved successfully at: {LOCAL_PATH}")