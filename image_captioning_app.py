import gradio as gr
import numpy as np
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def caption_image(input_image: np.ndarray):
    """
    Generate a caption for the input image.
    """
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    inputs = processor(images=raw_image, return_tensors="pt").to("cpu")  # Adjust device if necessary

    # Generate a caption
    outputs = model.generate(**inputs)
    
    # Decode the generated tokens to text
    caption = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return caption
iface = gr.Interface(
    fn=caption_image, 
    inputs=gr.Image(), 
    outputs="text",
    title="Image Captioning",
    description="Upload an image and get an AI-generated caption using the BLIP-2 model."
)
iface.launch()
