import streamlit as st
from PIL import Image
from pathlib import Path

from experiment import Experiment
from analysis import Analysis

favicon_path = Path.cwd()/'assets/intelchain_square.png'
im = Image.open(favicon_path)


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
        
# st.write(st.session_state)

with st.sidebar:
    with st.expander("How to Cite"):
        st.text(
            "If you use this application in your research, please cite it as follows."
        )
        st.caption(citation_apa)
        st.code(citation_bibtext, language='latex', wrap_lines=True) 

cwd = Path.cwd()
# experiment_path = cwd/'src/experiment.py'
# analysis_path = cwd/'src/analysis.py'
# st.write(experiment_path)

experiment_page = st.Page(Experiment, title='Run Experiments', default=True)
analysis_page = st.Page(Analysis, title='Analyze Results')

pg = st.navigation([experiment_page, analysis_page])

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