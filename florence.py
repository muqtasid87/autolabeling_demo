from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
    pipeline,
    AutoModelForCausalLM)
from transformers.image_utils import load_image
import torch
from PIL import Image
from tqdm import tqdm
import os
import time
from codecarbon import EmissionsTracker
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import psutil
import numpy as np
import requests


#load model and processor
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_florence = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor_florence = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

# BBox detection using Florence
#inference function: input prompt, image_path
def grounding(image_path, text_input=None, task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"):
    with open(image_path, "rb") as f:
              image = Image.open(f).convert("RGB")

    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor_florence(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model_florence.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=2048,
      num_beams=3
    )
    generated_text = processor_florence.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor_florence.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    return parsed_answer[task_prompt]