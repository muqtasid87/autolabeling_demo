import streamlit as st
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor
)
import torch
from PIL import Image
import time
import os



@st.cache_resource
def load_model():
    """Load the model and processor (cached to prevent reloading)"""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4", 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4")
    return model, processor

def process_image(image, prompt, model, processor):
    """Process the image and return the model's output"""
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

def main():
    # Compact header
    st.markdown("<h1 style='font-size: 24px;'>🔍 Image Analysis with Qwen2-VL</h1>", unsafe_allow_html=True)

    # Load model and processor
    with st.spinner("Loading model... This might take a minute."):
        model, processor = load_model()

    # Initialize session state
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'result' not in st.session_state:
        st.session_state.result = None
    if 'inference_time' not in st.session_state:
        st.session_state.inference_time = None

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
    default_prompt = "What type of vehicle is this? Choose only from: car, pickup, bus, truck, motorbike, van. Answer only in one word."
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
                result, inference_time = process_image(image, prompt, model, processor)
                st.session_state.result = result
                st.session_state.inference_time = inference_time
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if st.session_state.result:
        st.success("Analysis Complete!")
        st.markdown(f"**Result:**\n{st.session_state.result}")
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