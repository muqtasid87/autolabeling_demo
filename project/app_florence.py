import streamlit as st
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor
)
import torch
from PIL import Image
import time
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np


@st.cache_resource
def load_model():
    """Load the model and processor (cached to prevent reloading)"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", 
        torch_dtype=torch_dtype, 
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large-ft", 
        trust_remote_code=True
    )
    return model, processor, device, torch_dtype

def draw_bounding_boxes(image, bboxes, labels):
    """Draw bounding boxes and labels on the image"""
    # Convert PIL image to numpy array
    img_array = np.array(image)
    
    # Create figure and axis
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    
    # Add each bounding box and label
    for bbox, label in zip(bboxes, labels):
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        
        # Create rectangle patch
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label above the box
        plt.text(
            x, y-5,
            label,
            color='red',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0)
        )
    
    # Remove axes
    plt.axis('off')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def process_image(image, text_input, model, processor, device, torch_dtype):
    """Process the image and return the model's output"""
    start_time = time.time()
    
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    prompt = task_prompt + text_input if text_input else task_prompt

    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt"
    ).to(device, torch_dtype)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=2048,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, 
        task=task_prompt, 
        image_size=(image.width, image.height)
    )

    inference_time = time.time() - start_time
    
    # Create annotated image
    result = parsed_answer[task_prompt]
    annotated_image = draw_bounding_boxes(
        image,
        result['bboxes'],
        result['labels']
    )
    
    return result, inference_time, annotated_image

def main():
    # Compact header
    st.markdown("<h1 style='font-size: 24px;'>🔍 Image Analysis with Florence-2</h1>", unsafe_allow_html=True)

    # Load model and processor
    with st.spinner("Loading model... This might take a minute."):
        model, processor, device, torch_dtype = load_model()

    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

    # Main content area
    col1, col2, col3 = st.columns([1, 1.5, 1])

    with col1:
        # Input method selection
        input_option = st.radio("Choose input method:", ["Use example image", "Upload image"], label_visibility="collapsed")
        
        if input_option == "Upload image":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            image_source = uploaded_file
            if uploaded_file:
                st.session_state.selected_image = uploaded_file
        else:
            image_source = st.session_state.selected_image

    # Default prompt and analysis section
    default_prompt = "What type of vehicle is this?"
    prompt = st.text_area("Enter prompt:", value=default_prompt, height=100)
    
    analyze_col1, analyze_col2 = st.columns([1, 2])
    with analyze_col1:
        analyze_button = st.button("Analyze Image", use_container_width=True, disabled=image_source is None)

    # Display selected image and results
    if image_source:
        try:
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
            st.image(image, caption="Selected Image", width=300)
        except Exception as e:
            st.error(f"Error loading image: {str(e)}")

    # Analysis results
    if analyze_button and image_source:
        with st.spinner("Analyzing..."):
            try:
                result, inference_time, annotated_image = process_image(image, prompt, model, processor, device, torch_dtype)
                st.session_state.result = result
                st.session_state.inference_time = inference_time
                st.session_state.annotated_image = annotated_image
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.result:
        st.success("Analysis Complete!")
        
        # Display the annotated image
        st.image(st.session_state.annotated_image, caption="Analyzed Image with Detections", use_container_width=True)
        
        # Display raw results and inference time
        st.markdown("**Raw Results:**")
        st.json(st.session_state.result)
        st.markdown(f"*Inference time: {st.session_state.inference_time:.2f} seconds*")

    # Example images section
    if input_option == "Use example image":
        st.markdown("### Example Images")
        example_images = [f for f in os.listdir("images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if example_images:
            # Create grid of images
            cols = st.columns(4)  # Adjust number of columns as needed
            for idx, img_name in enumerate(example_images):
                with cols[idx % 4]:
                    img_path = os.path.join("images", img_name)
                    img = Image.open(img_path)
                    img.thumbnail((150, 150))
                    
                    # Make image clickable
                    if st.button(
                        "📷",
                        key=f"img_{idx}",
                        help=img_name,
                        use_container_width=True
                    ):
                        st.session_state.selected_image = img_path
                        st.rerun()
                    
                    # Display image with conditional styling
                    st.image(
                        img,
                        caption=img_name,
                        use_container_width=True,
                    )
        else:
            st.error("No example images found in the 'images' directory")

if __name__ == "__main__":
    main() 