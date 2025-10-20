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

import itertools

from src.utils import SafeFormatter, NoAliasDumper
from src.utils import create_dataframe_from_results, process_uploaded_results_csv, convert_df_to_csv
from src.experiment import Experiment
from src.global_configs import ALL_MODELS, is_prod

favicon_path = Path.cwd()/'assets/intelchain_square.png'
im = Image.open(favicon_path)


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







# def parse_round_info(columns):
#     rounds_info = defaultdict(list)
#     filtered_columns = [col for col in columns if 'round' in col.lower()]
#     filtered_columns_round_number = [fc.split('_')[0] for fc in filtered_columns]
#     filtered_columns_number = [int(re.findall(r'\d+', fc)[0]) for fc in filtered_columns_round_number]
#     # total_rounds = max(filtered_columns_number) + 1
#     for number, column in zip(filtered_columns_number, filtered_columns):
#         rounds_info[number].append(column)
            
#     return rounds_info
    
# def analyze_choice(df, columns):
#     """Analyzes categorical choice data by counting occurrences."""
#     model_col = 'model_name'
    
#     split_columns = []
#     factor_cols = []
#     for c in columns:
#         if '_llm_response' in c: 
#             response_col = c
        
#         parts = c.split('_')
#         factor_index = parts[2]
#         try:
#             int(factor_index)
#         except:
#             continue
#         factor_name = parts[3]
    
#         split_columns.append((factor_index, factor_name))
#         factor_cols.append(c)
        
#     # st.write('split_columns', split_columns)
#     # st.write('factor_cols', factor_cols)
#     st.markdown(f"Counting `{response_col}` grouped by `{model_col}` and `{factor_cols}`.")

#     analysis_df = df.groupby([model_col, *factor_cols, response_col]).size().reset_index(name='count')
    
#     st.dataframe(analysis_df)

    
# def analyze_ranking(df, columns):
#     """Analyzes ranking data by calculating mean rank."""
#     model_col = 'model_name'
    
#     split_columns = []
#     factor_cols = []
#     for c in columns:
#         if '_llm_response' in c: 
#             response_col = c
        
#         parts = c.split('_')
#         # st.write('parts', parts)
#         factor_index = parts[2]
#         try:
#             int(factor_index)
#         except:
#             continue
#         factor_name = parts[3]
    
#         split_columns.append((factor_index, factor_name))
#         factor_cols.append(c)
    
    
#     st.markdown(f"Analyzing mean rank of `{response_col}` grouped by `{model_col}` and `{factor_cols}`.")
    
# # Cached function for the no-factors plot
# @st.cache_data(show_spinner="Generating scales plot (no factors)...", ttl=3600)
# def create_scales_no_factors_plot(analysis_df, model_col, ):
#     """Cache the bar plot for scales analysis when there are no factors."""
#     fig = px.bar(
#         analysis_df,
#         x=model_col,
#         y='mean',
#         error_y='std',
#         title="Mean Score by Model",
#         labels={'mean': 'Mean Score (with Std Dev)'}
#     )
#     return fig

# # Cached function for the first factors plot (by factor_col, colored by model)
# @st.cache_data(show_spinner="Generating scales plot (by factor)...", ttl=3600)
# def create_scales_factors_plot1(analysis_df, factor_col, model_col):
#     """Cache the first bar plot for scales analysis with factors."""
#     fig = px.bar(
#         analysis_df,
#         x=factor_col,
#         y='mean',
#         color=model_col,
#         barmode='group',
#         error_y='std',
#         title=f"Mean Score by {factor_col.split('_')[-1].title()} and Model",
#         labels={'mean': 'Mean Score (with Std Dev)', factor_col: factor_col.split('_')[-1].title()}
#     )
#     return fig

# # Cached function for the second factors plot (by model, colored by factor)
# @st.cache_data(show_spinner="Generating scales plot (by model)...", ttl=3600)
# def create_scales_factors_plot2(analysis_df, factor_col, model_col):
#     """Cache the second bar plot for scales analysis with factors."""
#     fig = px.bar(
#         analysis_df,
#         x=model_col,
#         y='mean',
#         color=factor_col,
#         barmode='group',
#         error_y='std',
#         title=f"Mean Score by {factor_col.split('_')[-1].title()} and Model",
#         labels={'mean': 'Mean Score (with Std Dev)', factor_col: factor_col.split('_')[-1].title()}
#     )
#     return fig



# model_to_country = {
#     "gemini-2.5-flash-lite":'American',
#     "gpt-4.1-nano": 'American',
#     "meta-llama/Llama-4-Scout-17B-16E-Instruct": 'American',
#     "deepseek-ai/DeepSeek-V3.2-Exp": 'Chinese',
#     "zai-org/GLM-4.6": 'Chinese',
#     "Qwen/Qwen3-Next-80B-A3B-Instruct": 'Chinese'
# }

# @st.cache_data()
# def analyze_CET(df):

    
#     model_col = 'model_name'
#     factor_cols = []
#     response_cols = []
#     for c in df.columns:
#         if '_llm_response' in c: 
#             response_cols.append(c)
            
#             continue
#         elif 'llm_raw' in c:
#             continue        
    
#         factor_cols.append(c)
        
#     st.write('before dropna:', df.shape[0])
#     analysis_df_source = df.copy()
#     # st.write(response_cols)
#     for response_col in response_cols:
#         analysis_df_source[response_col] = pd.to_numeric(analysis_df_source[response_col], errors='coerce')
#         analysis_df_source.dropna(subset=[response_col], inplace=True)
        
#     st.write('after dropna:', analysis_df_source.shape[0])
    
#     analysis_df_source['CETSCORE'] = analysis_df_source[response_cols].sum(1)
#     # st.write(analysis_df_source[model_col].unique().tolist())
#     block_var = 'round0_scales_nationality'
    
#     res = analysis_df_source[[model_col, block_var, 'CETSCORE']]
#     res['model_country'] = res[model_col].map(model_to_country)

#     # model_name
#     res_group = res.groupby([model_col, block_var])['CETSCORE'].agg(['mean', 'std']).reset_index()
#     res_group = res_group.round(2)
#     st.dataframe(res_group)
    
#     fig = cet_plot1(res_group, block_var)
#     plotly_config = dict(
#         width='stretch'
#     )
#     st.plotly_chart(fig, config=plotly_config, key='plotly_model_name_mean')
    
#     # model country
#     'No ''meta-llama/Llama-4-Scout-17B-16E-Instruct'
#     res_group_country = res[res['model_name']!='meta-llama/Llama-4-Scout-17B-16E-Instruct']
#     # st.dataframe(res_group_country)
#     res_group_country = res.groupby(['model_country', block_var])['CETSCORE'].agg(['mean', 'std']).reset_index()
#     res_group_country = res_group_country.round(2)
#     st.dataframe(res_group_country)
    
#     fig = cet_plot2(res_group_country, block_var)
#     plotly_config = dict(
#         width='stretch'
#     )
#     st.plotly_chart(fig, config=plotly_config, key='plotly_model_country_mean')
#     #  
    
#     res_group_text_list = []
#     for i, row in res_group.iterrows():
#         t = f"{model_to_country[row[model_col]]} Model, {row[model_col]}, {row[block_var].capitalize()} M = {row['mean']}, SD = {row['std']}"
#         res_group_text_list.append(t)
                
#     res_group_text = '\n\n'.join(res_group_text_list)   
#     st.markdown(f'''
# Shimp and Sharma (1987, p. 282) 
# show CETSCALE for the four geographic areas as 

# Detroit M = 68.58, SD = 25.96; 

# Carolinas M = 61.28, SD = 24.41; 

# Denver = 57.84, SD = 26.10;
 
# Los Angeles M = 56.62, SD = 26.37.

# For our models,
# we show CETSCALE for each model as follows,

# {res_group_text}


# `Shimp, T. A., & Sharma, S. (1987). Consumer ethnocentrism: Construction and validation of the CETSCALE. Journal of marketing research, 24(3), 280-289.`
# ''')

# @st.cache_data()
# def cet_plot1(res_group, block_var):
#     fig = px.bar(
#         res_group,
#         x='model_name',
#         y='mean',
#         color=block_var,
#         barmode='group',
#         error_y='std',
#         title="Mean Score by Model",
#         labels={'mean': 'Mean Score (with Std Dev)'}
#     )
#     return fig

# @st.cache_data()
# def cet_plot2(res_group_country, block_var):
#     fig = px.bar(
#             res_group_country,
#             x='model_country',
#             y='mean',
#             color=block_var,
#             barmode='group',
#             error_y='std',
#             title="Mean Score by Model",
#             labels={'mean': 'Mean Score (with Std Dev)'}
#     )

#     return fig

# @st.cache_data()
# def analyze_scales(df, columns):
#     """Analyzes Likert scale data by calculating mean and std dev."""
#     model_col = 'model_name'

#     factor_cols = []
#     for c in columns:
#         if '_llm_response' in c: 
#             response_col = c
#             continue
#         elif 'llm_raw' in c:
#             continue        
    
#         factor_cols.append(c)
    
        
#     if factor_cols:
#         st.markdown(f"Grouping by `{model_col}` and `{factor_cols}`. Aggregating `{response_col}`.")
#     else:
#         st.markdown(f"Grouping by `{model_col}`. Aggregating `{response_col}`.")
#         st.caption('*No Factors in this Round.*')

#     analysis_df_source = df.copy()
#     analysis_df_source[response_col] = pd.to_numeric(analysis_df_source[response_col], errors='coerce')
#     analysis_df_source.dropna(subset=[response_col], inplace=True)
    
        
#     # no factor in this scales round
#     if not factor_cols:
#         analysis_df = analysis_df_source.groupby([model_col])[response_col].agg(['mean', 'std']).reset_index()
#         analysis_df = analysis_df.round(2)
#         # st.dataframe(analysis_df)

#         # Use cached plot
#         fig = create_scales_no_factors_plot(analysis_df, model_col)
#         plotly_config = dict(width='stretch')
#         st.plotly_chart(fig, config=plotly_config)
#         return 

#     for factor_col in factor_cols:
#         analysis_df = analysis_df_source.groupby([model_col, factor_col])[response_col].agg(['mean', 'std']).reset_index()
#         analysis_df = analysis_df.round(2)
#         # st.dataframe(analysis_df)

#         # Use cached plots
#         fig1 = create_scales_factors_plot1(analysis_df, factor_col, model_col)
#         plotly_config = dict(width='stretch')
#         st.plotly_chart(fig1, config=plotly_config, key=f'plotly_model_country_mean_{factor_col}_1')
        
#         fig2 = create_scales_factors_plot2(analysis_df, factor_col, model_col)
#         st.plotly_chart(fig2, config=plotly_config, key=f'plotly_model_country_mean_{factor_col}_2')


# @st.cache_data(show_spinner="Running analysis...", ttl=3600)
# def run_analysis(df):
#         # st.write('# Analysis')
#         # st.write(df.iloc[:3].to_csv())
        
#         rounds_info = parse_round_info(df.columns)
#         if not rounds_info:
#             st.error("Could not find any columns matching the 'round<N>_<type>_<name>' pattern.")
#             return

#         # Sort rounds by number to ensure chronological order
#         for round_num, columns in rounds_info.items():
#             # st.write(round_num, columns)
#             one_column = columns[0]
#             round_type = one_column.split('_')[1]
#             # st.write(round_type)

#             to_expand = True if round_num==0 else False
#             with st.expander(f"### Round {round_num+1}: {round_type.capitalize()}", expanded=to_expand):
                
#                 # Filter out rows where the necessary columns for this round are NaN
#                 # This is important because not all rows have data for all rounds
#                 round_df = df[['model_name', 'iteration', *columns]].dropna()
#                 if round_df.empty:
#                     st.info(f"No data available for Round {round_num}.")
#                     continue

#                 # Dispatch to the correct analysis function based on type
#                 if round_type == 'scales':
#                     analyze_scales(round_df, columns)
#                 elif round_type == 'choice':
#                     analyze_choice(round_df, columns)
#                 elif round_type == 'ranking':
#                     analyze_ranking(round_df, columns)
    



# def show_results_analysis(df, selected_config_path):
#     # st.success("Experiment Run Complete.")

#     # st.write('---')
#     st.subheader(
#         'Results', 
#         anchor='results', 
#         # divider='grey',
#         )
#     with st.expander('Load Results'):
#         st.file_uploader(
#             "Upload a csv file for an experiment you already ran",
#             type=['csv'],
#             key="csv_results_uploader",  # A unique key is required for on_change
#             on_change=process_uploaded_results_csv # The callback function
#         )
#     st.dataframe(df.head(1000), width='stretch')
#     fn = Path(selected_config_path)
#     fn = f'{fn.stem}.csv'
#     csv_df = convert_df_to_csv(df)
#     st.download_button(
#         "Download results",
#         csv_df,
#         fn,
#         "text/csv",
#         key='download-csv',
#         width='stretch',
#     )

# def Analysis():
#     if st.session_state.get('results'):
#         # with Profiler():            
#             df = create_dataframe_from_results(st.session_state['results'])
#             selected_config_path = st.session_state.get('selected_config_path')
#             show_results_analysis(df, selected_config_path)
#             with st.expander('# Analysis', expanded=False):
#                 run_analysis(df)
                
            
#             if not is_prod:
#                 st.checkbox('Show CETSCALE Analysis', value=False, key='show_cetscale')

#                 if st.session_state.get('show_cetscale'):
#                     # st.write('showing CET')
#                     with st.expander('# CETSCALE', expanded=False):
#                         analyze_CET(df)


def main():

    with st.sidebar:
        with st.expander("How to Cite"):
            st.text(
                "If you use this application in your research, please cite it as follows."
            )
            st.caption(citation_apa)
            st.code(citation_bibtext, language='latex', wrap_lines=True) 

    # cwd = Path.cwd()
    # experiment_path = cwd/'src/experiment.py'
    # analysis_path = cwd/'src/analysis.py'
    # st.write(experiment_path)

    experiment_page = st.Page(Experiment, title='Run Experiments', default=True)
    # analysis_page = st.Page(Analysis, title='Analyze Results')

    pg = st.navigation([
        experiment_page, 
        # analysis_page,
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