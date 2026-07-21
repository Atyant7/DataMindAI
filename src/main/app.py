import streamlit as st

from src.backend.data_loader import load_data
from src.frontend.workspace import show_workspace
from src.frontend.sidebar import show_sidebar
from src.frontend.home import show_home
from src.core.app_state import app_state
from src.backend.dataset_intelligence import DatasetIntelligence

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
    
    #store dataset profile
    app_state.dataset_profile = intelligence.generate_profile()

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
        