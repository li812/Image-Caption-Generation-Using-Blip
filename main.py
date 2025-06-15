import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

# Check for GPU availability
device = "cpu" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

def caption_image(input_image: np.ndarray):
    try:
        # Validate input
        if input_image is None:
            return "Error: No image provided"
        
        raw_image = Image.fromarray(input_image).convert('RGB')
        
        # Resize large images for better performance
        if max(raw_image.size) > 1024:
            raw_image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        
        text = "a photograph of"
        inputs = processor(raw_image, text, return_tensors="pt").to(device)
        
        # Enhanced generation parameters
        outputs = model.generate(
            **inputs, 
            max_length=75,
            num_beams=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.7
        )
        
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
        
    except Exception as e:
        return f"Error processing image: {str(e)}"

iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a descriptive caption using the BLIP model. Supports common image formats (PNG, JPG, WebP).",
    examples=None  # Add example images here
)

if __name__ == "__main__":
    iface.launch(share=False, server_name="127.0.0.1")