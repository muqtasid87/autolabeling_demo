#!/bin/bash
# Install PyTorch first
pip install -r requirements-torch.txt

# Wait for PyTorch to be fully installed
python -c "import torch" || exit 1

# Install remaining dependencies
pip install -r requirements.txt

# Start Streamlit app
streamlit run app_master.py