import streamlit as st 

def show_sidebar():
    st.sidebar.title("DataMindAI")
    
    page = st.sidebar.radio(
        "Navigations",
        [
            "Dashboard",
            "Visualization",
            "Data Cleaning",
            "Machine Learning",
            "AI Chat",₹
            "Report"
        ]
    )