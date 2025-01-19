import streamlit as st
from transformers import (
    Qwen2VLForConditionalGeneration,
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
def load_models():
    """Load both models and processors"""
    # Load Qwen model
    qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
    qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")

    # Load Florence model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    florence_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large-ft", 
        torch_dtype=torch_dtype, 
        trust_remote_code=True
    ).to(device)
    florence_processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-large-ft", 
        trust_remote_code=True
    )

    return qwen_model, qwen_processor, florence_model, florence_processor, device, torch_dtype

def process_qwen(image, prompt, model, processor):
    """Process image with Qwen2-VL"""
    start_time = time.time()
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt").to("cuda")

    output_ids = model.generate(**inputs, max_new_tokens=100)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    inference_time = time.time() - start_time
    return output_text[0].strip(), inference_time

def draw_bounding_boxes(image, bboxes, labels):
    """Draw bounding boxes and labels on the image"""
    img_array = np.array(image)
    fig, ax = plt.subplots()
    ax.imshow(img_array)
    
    for bbox, label in zip(bboxes, labels):
        x, y, x2, y2 = bbox
        width = x2 - x
        height = y2 - y
        
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=2,
            edgecolor='red',
            facecolor='none'
        )
        ax.add_patch(rect)
        
        plt.text(
            x, y-5,
            label,
            color='red',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0)
        )
    
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    return Image.open(buf)

def process_florence(image, text_input, model, processor, device, torch_dtype):
    """Process image with Florence-2"""
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
    result = parsed_answer[task_prompt]
    annotated_image = draw_bounding_boxes(
        image,
        result['bboxes'],
        result['labels']
    )
    
    return result, inference_time, annotated_image

def main():
    st.markdown("<h1 style='font-size: 24px;'>🚗 Vehicle Analysis Pipeline</h1>", unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models... This might take a minute."):
        qwen_model, qwen_processor, florence_model, florence_processor, device, torch_dtype = load_models()

    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'qwen_result' not in st.session_state:
        st.session_state.qwen_result = None
    if 'florence_result' not in st.session_state:
        st.session_state.florence_result = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None

    # Image selection
    col1, col2 = st.columns([1, 2])

    with col1:
        input_option = st.radio("Choose input method:", ["Use example image", "Upload image"], label_visibility="collapsed")
        
        if input_option == "Upload image":
            uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            image_source = uploaded_file
            if uploaded_file:
                st.session_state.selected_image = uploaded_file
        else:
            image_source = st.session_state.selected_image

        # Default prompt for Qwen
        default_prompt = "What type of vehicle is this? Choose only from: car, pickup, bus, truck, motorbike, van. Answer only in one word."
        prompt = st.text_area("Enter prompt for classification:", value=default_prompt, height=100)
        
        analyze_button = st.button("Analyze Image", use_container_width=True, disabled=image_source is None)

    # Display and process
    if image_source:
        try:
            if isinstance(image_source, str):
                image = Image.open(image_source).convert("RGB")
            else:
                image = Image.open(image_source).convert("RGB")
            
            with col2:
                st.image(image, caption="Selected Image", width=300)

            if analyze_button:
                # Step 1: Qwen Analysis
                with st.spinner("Step 1: Classifying vehicle type..."):
                    qwen_result, qwen_time = process_qwen(image, prompt, qwen_model, qwen_processor)
                    st.session_state.qwen_result = qwen_result
                
                # Step 2: Florence Analysis
                with st.spinner("Step 2: Detecting vehicle location..."):
                    florence_result, florence_time, annotated_image = process_florence(
                        image, 
                        f"Find the {qwen_result} in the image", 
                        florence_model, 
                        florence_processor, 
                        device, 
                        torch_dtype
                    )
                    st.session_state.florence_result = florence_result
                    st.session_state.annotated_image = annotated_image

                # Display results
                st.markdown("### Analysis Results")
                
                # Qwen results
                st.markdown("#### Step 1: Vehicle Classification")
                st.markdown(f"**Type:** {st.session_state.qwen_result}")
                st.markdown(f"*Classification time: {qwen_time:.2f} seconds*")
                
                # Florence results
                st.markdown("#### Step 2: Vehicle Detection")
                st.image(annotated_image, caption="Vehicle Detection Result", use_container_width=True)
                st.markdown(f"*Detection time: {florence_time:.2f} seconds*")
                st.markdown("**Raw Detection Data:**")
                st.json(florence_result)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

    # Example images section
    if input_option == "Use example image":
        st.markdown("### Example Images")
        example_images = [f for f in os.listdir("images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if example_images:
            cols = st.columns(4)
            for idx, img_name in enumerate(example_images):
                with cols[idx % 4]:
                    img_path = os.path.join("images", img_name)
                    img = Image.open(img_path)
                    img.thumbnail((150, 150))
                    
                    if st.button("📷", key=f"img_{idx}", help=img_name, use_container_width=True):
                        st.session_state.selected_image = img_path
                        st.rerun()
                    
                    st.image(img, caption=img_name, use_container_width=True)
        else:
            st.error("No example images found in the 'images' directory")

if __name__ == "__main__":
    main() 