import streamlit as st 
from src.core.config import SUPPORTED_FILE_TYPES

def show_sidebar():
    st.sidebar.title("DataMindAI")
    
    st.sidebar.divider()
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload Dataset",
        type=SUPPORTED_FILE_TYPES
    )
    
    st.sidebar.divider()
    
    st.sidebar.subheader("📊 Dataset Information")

    st.sidebar.info(
        "No dataset uploaded."
    )

    st.sidebar.divider()

    if st.sidebar.button("💬 New Chat"):

        st.sidebar.success("New Chat Started!")

    if st.sidebar.button("🗑 Clear Session"):

        st.sidebar.warning("Session Cleared!")

    st.sidebar.divider()

    st.sidebar.subheader("⚙ Settings")

    st.sidebar.write("Coming Soon...")

    return uploaded_file