import streamlit as st

from .analysis_component import analysis_main

analysis_page = st.Page(analysis_main, title='Analyze Results')
