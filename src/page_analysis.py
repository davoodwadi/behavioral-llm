import streamlit as st

from .analysis_component import analysis_component

analysis_page = st.Page(analysis_component, title='Analyze Results')
