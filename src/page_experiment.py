import streamlit as st

from experiment import Experiment

experiment_page = st.Page(Experiment, title='Run Experiments', default=True)
