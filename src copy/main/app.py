import streamlit as st

from src.backend.data_loader import load_data
from src.frontend.workspace import show_workspace
from src.frontend.sidebar import show_sidebar
from src.frontend.home import show_home
from src.core.app_state import app_state
from src.backend.dataset_intelligence import DatasetIntelligence
from src.backend.preprocessing_intelligence import PreprocessingIntelligence
from src.frontend.preprocessing_workspace import show_processing_workspace

def initialize_application():
    st.set_page_config(
        page_title="DataMindAI",
        page_icon="🧠",
        layout="wide"
    )

def load_dataset(uploaded_file):
    #Load dataframe
    dataframe = load_data(uploaded_file)
    
    #Store Dataframe
    app_state.dataset = dataframe
    app_state.dataset_name = uploaded_file.name
    
    #Analyze dataset
    intelligence = DatasetIntelligence( dataframe, uploaded_file.name )
    
    profile = intelligence.generate_profile()
    
    #store dataset profile
    app_state.dataset_profile = profile
    
    # ---------------------------------------------
    # Preprocessing Intelligence
    # ---------------------------------------------
    preprocessing = PreprocessingIntelligence(
        dataframe,
        profile
    )

    app_state.preprocessing_plan = (
        preprocessing.generate_plan()
    )

def main():

    initialize_application()
    
    uploaded_file, page = show_sidebar()

    # ---------------------------------------------
    # No Dataset Uploaded
    # ---------------------------------------------
    if not uploaded_file:
        show_home()
        return

    # ---------------------------------------------
    # Load Dataset Only Once
    # ---------------------------------------------
    if (
        not app_state.has_dataset()
        or app_state.dataset_name != uploaded_file.name
    ):
        load_dataset(uploaded_file)

    # ---------------------------------------------
    # Workspace Navigation
    # ---------------------------------------------
    if page == "Dataset Intelligence":
        pass

    elif page == "Preprocessing Intelligence":
        pass
        
if __name__ == "__main__":
    main()
        