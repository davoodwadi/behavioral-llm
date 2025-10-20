import streamlit as st
from pathlib import Path

from PIL import Image

import pandas as pd
import plotly.express as px

from pathlib import Path
import yaml
import json
import random
import math
import numpy as np

import uuid
from io import StringIO

from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Optional

import os 
import re
import time
from datetime import datetime

import sys

import itertools

import psutil
from pympler import asizeof

from src.utils import SafeFormatter, NoAliasDumper
from src.utils import create_dataframe_from_results, process_uploaded_results_csv, convert_df_to_csv
from src.global_configs import ALL_MODELS, is_prod

from src.page_analysis import analysis_page
from src. page_experiment import experiment_page

favicon_path = Path.cwd()/'assets/intelchain_square.png'
im = Image.open(favicon_path)

# Example: get total memory used by all items in session_state
# total_bytes = sum(sys.getsizeof(v) for v in st.session_state.values())
# st.write(f"Total size of session_state (shallow): {total_bytes / 1024:.2f} KB")
# process = psutil.Process(os.getpid())
# mem_info = process.memory_info()
# st.write(f"RSS memory: {mem_info.rss / (1024**2):.2f} MB")
# st.write(st.session_state)

# Get the path to the 'config' directory (assuming it's relative to the app file)
APP_FILE = Path(__file__)
APP_DIR = APP_FILE.parent
PROJECT_DIR = APP_DIR.parent
CONFIG_DIR = PROJECT_DIR / 'config'

def initialize_session_state():
    # App state flags
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    if 'last_uploaded_file' not in st.session_state:
        st.session_state.last_uploaded_file = None
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False
    if 'results' not in st.session_state:
        st.session_state.results = []
    # Default values for experiment settings
    if 'system_prompt' not in st.session_state:
        st.session_state.system_prompt = "Default prompt..."
    if 'models_to_test' not in st.session_state:
        st.session_state.models_to_test = []
    
    # Initialize API keys dictionary based on models from config
    if 'api_keys' not in st.session_state:
        st.session_state['api_keys'] = {m.split('-')[0]:None for m in ALL_MODELS}
    
initialize_session_state()

citation_bibtext = """@inproceedings{
wadi2025a,
title={A Monte-Carlo Sampling Framework For Reliable Evaluation of Large Language Models Using Behavioral Analysis},
author={Davood Wadi and Marc Fredette},
booktitle={The 2025 Conference on Empirical Methods in Natural Language Processing},
year={2025},
url={https://openreview.net/forum?id=lpm6tSIoE6}
}"""
citation_apa = '''Davood Wadi, & Marc Fredette (2025). A Monte-Carlo Sampling Framework For Reliable Evaluation of Large Language Models Using Behavioral Analysis. In The 2025 Conference on Empirical Methods in Natural Language Processing.'''

session_variable_names = [    
    'system_prompt',
    'rounds',
    'block_variables',
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'randomize',
    'paper_url',
    'api_keys',
    'selected_config_path',
]
for var in session_variable_names:
    if var in st.session_state:
        st.session_state[var] = st.session_state[var]


def monitor_mem():
    st.title("🧠 Memory Monitor")

    # Process reference
    process = psutil.Process(os.getpid())

    # Initialize tracking dataframe in session_state
    if "mem_history" not in st.session_state:
        st.session_state.mem_history = pd.DataFrame(columns=["time", "process_mb", "session_mb"])
        st.session_state.start_time = time.time()

    # Measure memory usage
    process_mem = process.memory_info().rss / 1024**2
    session_mem = asizeof.asizeof(st.session_state) / 1024**2
    elapsed = time.time() - st.session_state.start_time

    # Append reading to history
    new_row = pd.DataFrame([{
        "time": elapsed,
        "process_mb": process_mem,
        "session_mb": session_mem
    }])
    st.session_state.mem_history = pd.concat(
        [st.session_state.mem_history, new_row],
        ignore_index=True
    ).tail(1000)  # keep last 1000 readings

    # Plot
    st.line_chart(st.session_state.mem_history.set_index("time"))

    st.write(f"**Process memory:** {process_mem:.2f} MB")
    st.write(f"**Session state memory:** {session_mem:.2f} MB")

    st.caption("This updates each time the page reruns. To observe growth, interact with widgets or refresh.")
def main():
    monitor_mem()
    with st.sidebar:
        with st.expander("How to Cite"):
            st.text(
                "If you use this application in your research, please cite it as follows."
            )
            st.caption(citation_apa)
            st.code(citation_bibtext, language='latex', wrap_lines=True) 


    pg = st.navigation([
        experiment_page, 
        analysis_page,
        ])

    st.set_page_config(
        page_title="Behavioral LLM Experiment Hub",
        page_icon=im,
        layout="wide",
        initial_sidebar_state="collapsed"
    )


    pg.run()


    st.write('')
    st.write('')
    st.write('')
    st.write('')
    # if is_prod:
    with st.container(border=False):
        st.subheader("Cite This App", divider='grey', anchor='cite')
        # st.write('---')
        st.text("Please use the following citation if you use this app in your work.")
        st.caption(citation_apa)
        st.code(citation_bibtext, language='latex')
    
    
if __name__=='__main__':
    main()