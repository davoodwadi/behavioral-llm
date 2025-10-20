import streamlit as st

from .analysis import Analysis

analysis_page = st.Page(Analysis, title='Analyze Results')
