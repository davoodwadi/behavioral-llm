import streamlit as st

from .experiment_component import experiment_main

experiment_page = st.Page(experiment_main, title='Run Experiments', default=True)
