import streamlit as st
import app_qwen
import project.app_florence as app_florence
import project.app_combined as app_combined

# Set page configuration
st.set_page_config(
    page_title="Vehicle Analysis Suite",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"  # Show sidebar by default
)

# Custom CSS for the sidebar and main content
st.markdown("""
    <style>
        .block-container {padding-top: 1rem; padding-bottom: 0rem;}
        .element-container {margin-bottom: 0.5rem;}
        .stButton button {width: 100%;}
        h1 {margin-bottom: 1rem;}
        .sidebar-content {
            padding: 1rem;
        }
        .app-header {
            text-align: center;
            padding: 1rem;
            background-color: #f0f2f6;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

def main():
    # Sidebar for app selection
    with st.sidebar:
        st.markdown("### 🚗 Vehicle Analysis Suite")
        st.markdown("---")
        app_mode = st.radio(
            "Select Analysis Mode:",
            ["Qwen2-VL Classifier", "Florence-2 Detector", "Combined Pipeline"],
            index=0,  # Default to Qwen2-VL
            key="app_selection"
        )
        
        st.markdown("---")
        st.markdown("""
        ### About the Models:
        
        **Qwen2-VL Classifier**
        - Quick vehicle classification
        - Single-word output
        - Optimized for vehicle types
        
        **Florence-2 Detector**
        - Visual object detection
        - Bounding box visualization
        - Detailed spatial analysis
        
        **Combined Pipeline**
        - Two-stage analysis
        - Classification + Detection
        - Comprehensive results
        """)

    # Clear previous app states when switching
    if 'last_app' not in st.session_state:
        st.session_state.last_app = None
    
    if st.session_state.last_app != app_mode:
        # Clear relevant session state variables
        for key in list(st.session_state.keys()):
            if key not in ['app_selection', 'last_app']:
                del st.session_state[key]
        st.session_state.last_app = app_mode

    # Main content area
    if app_mode == "Qwen2-VL Classifier":
        st.markdown("""
            <div class='app-header'>
                <h1>🤖 Qwen2-VL Vehicle Classifier</h1>
                <p>Specialized in quick and accurate vehicle type classification</p>
            </div>
        """, unsafe_allow_html=True)
        app_qwen.main()

    elif app_mode == "Florence-2 Detector":
        st.markdown("""
            <div class='app-header'>
                <h1>🔍 Florence-2 Vehicle Detector</h1>
                <p>Advanced visual detection with bounding box visualization</p>
            </div>
        """, unsafe_allow_html=True)
        app_florence.main()

    else:  # Combined Pipeline
        st.markdown("""
            <div class='app-header'>
                <h1>🚀 Combined Analysis Pipeline</h1>
                <p>Comprehensive vehicle analysis using both models</p>
            </div>
        """, unsafe_allow_html=True)
        app_combined.main()

if __name__ == "__main__":
    main() 