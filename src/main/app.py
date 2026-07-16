import streamlit as st

from src.backend.data_loader import load_data
from src.frontend.workspace import show_workspace
from src.frontend.sidebar import show_sidebar
from src.frontend.home import show_home
from src.core.app_state import app_state

def initialize_application():
    st.set_page_config(
        page_title="DataMindAI",
        page_icon="🧠",
        layout="wide"
    )

def load_dataset(uploaded_file):
    app_state.dataset = load_data(uploaded_file)
    app_state.dataset_name = uploaded_file.name

def main():

    initialize_application()
    
    uploaded_file = show_sidebar()


    if uploaded_file:
        
        if (not app_state.has_dataset() or app_state.dataset_name != uploaded_file.name):
            load_dataset(uploaded_file)
            show_workspace()
        
    else:
        show_home()
        
if __name__ == "__main__":
    main()
        