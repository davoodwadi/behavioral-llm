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

from src.utils import get_llm_text_mixed_rank, get_llm_text_mixed_choice, get_llm_text_mixed_scales, get_segment_text_from_segments_list, SafeFormatter, create_dataframe_from_results, process_uploaded_results_csv, convert_df_to_csv
from src.models import get_model, LLMResponse
from src.experiment import Experiment


class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True
 

favicon_path = Path.cwd()/'assets/intelchain_square.png'
im = Image.open(favicon_path)


APP_ENVIRONMENT = os.environ.get('APP_ENV', 'dev') 

is_prod = (APP_ENVIRONMENT == 'production')

# Get the path to the 'config' directory (assuming it's relative to the app file)
APP_FILE = Path(__file__)
APP_DIR = APP_FILE.parent
PROJECT_DIR = APP_DIR.parent
CONFIG_DIR = PROJECT_DIR / 'config'
EXPERIMENT_TYPES = ['mixed', 'choice', 'scales', 'ranking']

ALL_MODELS = [  
    "openai-gpt-4.1-nano",
    "anthropic-claude-3-5-haiku-20241022",
    "google-gemini-2.5-flash-lite",
    "google-gemini-2.5-flash",
    "google-gemini-2.5-pro",
    "deepinfra-Qwen/Qwen3-Next-80B-A3B-Instruct",
    "deepinfra-deepseek-ai/DeepSeek-V3.2-Exp",
    "deepinfra-deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepinfra-zai-org/GLM-4.6",
    "deepinfra-meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepinfra-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepinfra-moonshotai/Kimi-K2-Instruct-0905",
    "alibabacloud-qwen3-max",
    "alibabacloud-qwen-plus",
    "alibabacloud-qwen-flash",
]
if not is_prod:
    ALL_MODELS.append('test-test')

config_paths = {
    'choice': 'config/choice/COO/country of origin choice.yaml',
    'scales': 'config/scales/ad_framing_and_scarcity.yaml',
    'ranking': 'config/rank/Balabanis 2004/Domestic Country Bias.yaml',
    'mixed': 'config/mixed/Domestic Country Bias.yaml',
}
Round_Types = {
    'scales': {
        "Segment_Types": ['Fixed Segment', 'Treatment Segment']
    },
    'choice': {
        "Segment_Types": ['Fixed Segment', 'Choice Segment']
    },
    'ranking': {
        "Segment_Types": ['Fixed Segment', 'Ranking Segment']
    }
}
        
factors_to_save = [
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
    'selected_config_path'
]


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




# def process_uploaded_yaml():
#     """
#     This function is called ONLY when the file_uploader's state changes.
#     It reads the uploaded file and updates the session state.
#     """
#     # Get the uploaded file from session state using the key.
#     uploaded_file = st.session_state.yaml_uploader
    
#     if uploaded_file is not None:
#         try:
#             # To read the file as a string
#             stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#             string_data = stringio.read()
#             yaml_data = yaml.safe_load(string_data)
#             # store the config in state for reset_config
#             st.session_state['yaml_data'] = yaml_data
#             # Update session state with the keys from the YAML file
#             for state_var_name in yaml_data.keys():
#                 st.session_state[state_var_name] = yaml_data.get(state_var_name)
            
#             # reset selected_round and selected_round_counter
#             if yaml_data.get('rounds'):
#                 st.session_state.selected_round = st.session_state.rounds[0]
#                 st.session_state.selected_round_counter = 0

#             if yaml_data.get('api_keys'):
#                 api_keys = yaml_data.get('api_keys')
#                 for provider_name, api_key in api_keys.items():
#                     st.session_state[f'{provider_name}_api_key'] = api_key

#             st.success(f"Successfully loaded configuration from '{uploaded_file.name}'!")
        
#         except Exception as e:
#             st.error(f"Error processing YAML file: {e}")

# def show_experiment_configs_selectors():
#     with st.expander('Experiment Configs'):
#         k = st.session_state.get('k', 1)
#         st.number_input(
#             "Number of Iterations", 
#             value=k, 
#             placeholder="Type a number...",
#             key = 'k',
#         )

#         test = st.session_state.get('test', True)
#         st.toggle('Run Mock Experiments', value=test, key='test')
        
#         sleep_amount = st.session_state.get('sleep_amount', 0.1)
#         st.number_input(
#             'Amount to pause between LLM API calls (seconds)', 
#             value=sleep_amount,
#             key = 'sleep_amount'
#         )

#         models_to_test = st.session_state.get('models_to_test', [])
#         filtered_models_to_test = [m for m in models_to_test if m in ALL_MODELS]
#         if not filtered_models_to_test:
#             filtered_models_to_test = ALL_MODELS[0]
#         st.multiselect(
#             'LLMs to Test', 
#             ALL_MODELS, 
#             default=filtered_models_to_test,
#             key='models_to_test'
#         )

#         # Based on st.session_state.models_to_test
#         # get required api keys
#         if st.session_state.get('api_keys'):
#             api_keys = st.session_state.get('api_keys')
#         else:
#             api_keys = dict()

#         if not st.session_state.test:
#             selected_providers = set([m.split('-')[0] for m in st.session_state.models_to_test])
#             for provider_name in selected_providers:
#                 the_key = api_keys.get(provider_name, '')

#                 st.text_input(
#                     f'{provider_name.capitalize()} API key',
#                     value=the_key,
#                     type="password", 
#                     key=f'{provider_name}_api_key'
#                 )

#                 the_widget_key = st.session_state.get(f'{provider_name}_api_key')
#                 st.session_state['api_keys'][provider_name] = the_widget_key
                        
#         # 

#         paper_url = st.session_state.get('paper_url', '')
#         st.text_input(
#             "Paper URL (optional)", 
#             value=paper_url,
#             key='paper_url'
#         )

#         randomize = st.session_state.get('randomize', True)
#         st.checkbox(
#             'Randomize Items Displayed in Ranking/Choice Segments', 
#             randomize,
#             key='randomize'
#         )

# default_config = {
#     'system_prompt':'',
#     'rounds':[],
#     'block_variables':[],
#     'k':1, 
#     'test':True, 
#     'sleep_amount':0.01,
#     'models_to_test':[],
#     'randomize':True,
#     'paper_url':'',
#     'api_keys':None
# }
# def reset_config():
#     for factor_to_load in default_config.keys():
#         st.session_state[factor_to_load] = default_config.get(factor_to_load)
#     if st.session_state.rounds:
#         st.session_state.selected_round = st.session_state.rounds[0]
#         st.session_state.selected_round_counter = 0
#     else:
#         st.session_state.selected_round = None
#         st.session_state.selected_round_counter = None    

#     if default_config.get('api_keys'):
#         api_keys = default_config.get('api_keys')
#         for provider_name, api_key in api_keys.items():
#             st.session_state[f'{provider_name}_api_key'] = api_key

#     st.rerun()
    

# @st.dialog("Add Block Variable")
# def show_add_block_variable_dialog():
#     df = pd.DataFrame({
#         'Level Name': ["Name 1", "Name 2"],
#         'Level Text for LLM': ["Text 1", "Text 2"],
#         'Group ID': ["", ""],
#     })
#     new_factor_name = st.text_input('New Block Variable Name', key='new_block_variable_name_text_input')
#     edited_df = st.data_editor(
#     df, 
#     num_rows="dynamic",
#     key='data_editor_block_variable'
#     )
#     new_factor_names_levels = edited_df['Level Name'].values.tolist()
#     new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
#     new_factor_group_id = edited_df['Group ID'].values.tolist()
#     # st.write(new_factor_group_id)
#     new_fact = [
#         new_factor_name, 
#         [{'name':nf_name, 'text':nf_text, 'group_id':nf_group_id, 'factor_name': new_factor_name} for nf_name, nf_text, nf_group_id in zip(new_factor_names_levels, new_factor_text_levels, new_factor_group_id)]
#     ]
#     if st.button("Add Block Variable", key='add_button_block_variable'):
#         st.session_state['block_variables'].append(new_fact)

#         st.session_state.show_toast = True
#         st.session_state.toast_text = f'Block Variable Added: {new_factor_name}'

#         st.rerun() 

# def remove_block_variable(variable_name):
#     if st.session_state.get('block_variables'):
#         st.session_state['block_variables'] = [bvar for bvar in st.session_state['block_variables'] if bvar[0]!=variable_name]
#         st.toast(f'Block Variable: {variable_name} Removed.' )

# def show_block_variables():
#     with st.expander('Block Variables (optional)'):
#         st.info('Only one block variable is allowed here. Add more variables to indicate different variations of the same variable (e.g., country and nationality). For multiple variables use `group_id` to connect levels to each other.')
#         if st.session_state.get('block_variables'):
#             for factor_tuple in (st.session_state['block_variables']):
#                 factor_name = factor_tuple[0] # product
#                 factor_levels = factor_tuple[1] # [{name:'Name 1', text:'Text 1', factor_name:'product'},{product|name:'Name 2', product|text:'Text 2'}]
#                 col_factor, col_remove = st.columns([2,1])
#                 with col_factor:
#                     st.write(f'***{factor_name}***')
#                 with col_remove:
#                     if st.button(f'Remove {factor_name}', width='stretch', key=f"remove_button_block_variables_{factor_name}"):
#                         remove_block_variable(factor_name)
#                         st.rerun()
                
#                 original_df = pd.DataFrame(factor_levels)
#                 # st.write(factor_levels)
                
#                 editor_key = f"data_editor_block_variables_{factor_name}"

#                 st.data_editor(
#                     original_df, 
#                     num_rows='dynamic', 
#                     key=editor_key,
#                     on_change=apply_editor_changes_and_update_state,
#                     kwargs={
#                             "factor_name": factor_name,
#                             "state_list_to_update": st.session_state['block_variables'],
#                             "editor_key": editor_key,
#                             "original_df": original_df,  # Pass the original state to the callback
#                             "has_group_id": True,  # Pass the original state to the callback
#                         },
#                     width='stretch',
#                     column_order=("name", "text", 'group_id'),
#                 )

#         else:
#             st.session_state.block_variables = []
#             st.info('No Block Variables. You can optionally add them here.')

#         if st.button('Add Block Variable'):
#             show_add_block_variable_dialog()

# def apply_editor_changes_and_update_state(factor_name, editor_key, original_df, state_list_to_update, has_group_id=False):
#     """
#     Callback to apply the delta from st.data_editor to the original DataFrame
#     and then update the main application state.
#     This function is robust against empty/NaN rows.
#     """
#     # 1. Get the delta object from session_state. This is the source of truth for changes.
#     try:
#         delta = st.session_state[editor_key]
#     except KeyError:
#         # This can happen in rare edge cases, so we exit gracefully.
#         return

#     # Start with a clean copy of the data that was originally passed to the editor
#     final_df = original_df.copy()

#     # 2. Apply DELETED rows first
#     # delta["deleted_rows"] is a list of original integer indices to delete
#     if delta["deleted_rows"]:
#         # The `errors='ignore'` flag prevents crashes if an index is already gone
#         final_df = final_df.drop(index=delta["deleted_rows"], errors='ignore')

#     # 3. Apply ADDED rows
#     # delta["added_rows"] is a list of new row dictionaries
#     if delta["added_rows"]:
#         added_df = pd.DataFrame(delta["added_rows"])
#         final_df = pd.concat([final_df, added_df], ignore_index=True)
        
#     # 4. Apply EDITED rows last
#     # delta["edited_rows"] is a dict like {row_index: {col_name: new_value}}
#     for row_index_str, changed_data in delta["edited_rows"].items():
#         row_index = int(row_index_str)
#         for col_name, new_value in changed_data.items():
#             # Use .at for fast, label-based single-cell assignment
#             final_df.at[row_index, col_name] = new_value

#     # 5. --- CRITICAL DATA CLEANING STEP ---
#     final_df.fillna('', inplace=True)

#     final_df.reset_index(drop=True, inplace=True)

#     # 6. Convert the clean, final DataFrame back into your required state format
#     final_records = final_df.to_dict(orient='records')
    
#     if has_group_id:
#         formatted_levels = [
#             {"name": level.get('name', ''), "text": level.get('text', ''), "group_id": level.get('group_id', ''), 'factor_name': factor_name} for level in final_records
#         ]
#     else:
#         formatted_levels = [
#             {"name": level.get('name', ''), "text": level.get('text', ''), 'factor_name': factor_name}
#             for level in final_records
#         ]
        
#     new_factor_data = [factor_name, formatted_levels]

#     # 7. Find and update the factor in the main state object
#     try:
#         index = [i for i, fac in enumerate(state_list_to_update) if fac[0] == factor_name][0]
#         state_list_to_update[index] = new_factor_data
#     except IndexError:
#         st.error(f"Could not find factor '{factor_name}' to update.")


# def show_mixed_segments(current_round, round_counter):
#     factor_info = 'Use `{factor_name}` to add placeholders for factors\n\nExample: {product} or {price}\n\n'
#     round_type = current_round['round_type']
    
#     st.info(factor_info)
    
#     with st.container(border=True):
        
#         factor_names_raw = [f[0] for f in current_round['factors_list']]
#         block_variables = st.session_state.get('block_variables')
        
#         if block_variables:
#             factor_names_raw.extend([f[0] for f in block_variables])
        
#         factor_names_display = ['`{' + f + '}`' for f in factor_names_raw]
#         factor_names_str = ' '.join(factor_names_display)
        
#         segment_label = f"Factors to use: {factor_names_str}\n\n"
#         segments = current_round.get('segments')
        
        
#         for i, seg in enumerate(segments):
#             col1, col2 = st.columns([3,1])
#             with col1:
#                 st.write('Round Segment')
#             with col2:
#                 if round_type!='scales':
#                     if st.button('Remove Segment', key=f"remove_segment_{current_round['key']}_{seg['segment_id']}"):
#                         removed_segment = segments.pop(i)
#                         st.rerun()                        
                
#             text_content  = st.text_area(
#             segment_label, 
#             value=seg['segment_text'], 
#             key=f"segment_text_{current_round['key']}_{seg['segment_id']}"
#             )
#             current_round['segments'][i]['segment_text'] = text_content    

        
#         if (current_round['round_type'] == 'choice') or (current_round['round_type'] == 'ranking'):
#             if st.button('Add Segment', width='stretch'):
#                 # st.write('segment added')
#                 new_segment = {
#                     'segment_text':'',
#                     'segment_id': str(uuid.uuid4())
#                 }
#                 current_round['segments'].append(new_segment)
#                 st.session_state.show_toast = True
#                 st.session_state.toast_text = 'Segment Added'
#                 st.rerun()
#     # st.write('segments', segments)

# def get_config_to_save(factors_to_save):
#     # config = load_experiment_config(st.session_state.selected_config_path)
#     config_to_save = dict()
#     for factor_to_save in factors_to_save:
#         config_to_save[factor_to_save] = st.session_state.get(factor_to_save)
#         # st.write(st.session_state.get(factor_to_save))
#     return config_to_save

# @st.cache_data
# def get_rank_permutations_multi_factor(factors_list):
#     if not factors_list:
#         return []

#     factor_products = get_choice_factor_products(factors_list)
#     factor_levels_rank_permutations = list(itertools.permutations(factor_products, len(factor_products)))
#     factor_levels_rank_permutations_list = [list(permutation) for permutation in factor_levels_rank_permutations]
#     return factor_levels_rank_permutations_list



# @st.dialog('Sample User Message Shown to the LLM')
# def show_sample_mixed_modal(current_round, round_counter):
#     if current_round['round_type']=='ranking':
#         show_sample_ranking(current_round, round_counter)
#     elif current_round['round_type']=='choice':
#         show_sample_choice(current_round, round_counter)
#     elif current_round['round_type']=='scales':
#         show_sample_scales(current_round, round_counter)
       
       


# def choose_a_subset_of_combinations(all_combinations):
#     total_combinations = len(all_combinations)
    
#     selection_method = st.radio(
#     label="How would you like to select the combinations to run?",
#     options=("Run all combinations", "Select a random percentage", "Select a random count"),
#     index=0,  # Default to "Run all"
#     horizontal=True,
#     )

#     # This will hold the final list of combinations to be processed
#     combinations_to_run = []

#     if selection_method == "Run all combinations":
#         combinations_to_run = all_combinations
#         st.write(f"✅ All **{total_combinations}** combinations are selected.")

#     else:
#         # For both random options, we need a seed for reproducibility
#         seed = st.number_input("Enter a random seed for reproducibility", value=42, min_value=0)
#         random.seed(seed)

#         if selection_method == "Select a random percentage":
#             percentage = st.slider(
#                 "Select percentage of combinations to run:",
#                 min_value=1,
#                 max_value=100,
#                 value=10,
#                 format="%d%%"
#             )
#             count_to_select = int(total_combinations * (percentage / 100))
            
#             # Ensure at least one combination is selected if percentage > 0
#             count_to_select = max(1, count_to_select)
            
#             if total_combinations > 0:
#                 combinations_to_run = random.sample(all_combinations, k=count_to_select)
            
#             st.write(f"✅ **{count_to_select}** combinations ({percentage}%) selected randomly.")

#         elif selection_method == "Select a random count":
#             # Make sure the max value for the input is the total number of combinations
#             default_value = min(100, total_combinations)
#             count_to_select = st.number_input(
#                 "Enter the number of combinations to run:",
#                 min_value=1,
#                 max_value=total_combinations,
#                 value=default_value,
#                 step=10,
#             )
#             if total_combinations > 0:
#                 combinations_to_run = random.sample(all_combinations, k=int(count_to_select))
            
#             st.write(f"✅ **{len(combinations_to_run)}** combinations selected randomly.")
#     return combinations_to_run


# def show_results(df, selected_config_path):
#     st.success("Experiment Run Complete.")

#     # st.write('---')
#     st.subheader(
#         'Results', 
#         anchor='results', 
#         # divider='grey',
#         )
 
#     st.dataframe(df.head(1000), width='stretch')
#     fn = Path(selected_config_path)
#     fn = f'{fn.stem}.csv'
#     csv_df = convert_df_to_csv(df)
#     col_download, col_analyze = st.columns(2)
#     with col_download:
#         st.download_button(
#             "Download results",
#             csv_df,
#             fn,
#             "text/csv",
#             key='download-csv',
#             width='stretch',
#         )
#     with col_analyze:
#         if st.button('Analyze', width='stretch'):
#             st.switch_page('Analysis.py')
    
# def show_mixed_experiment_execution(combinations_to_run, selected_config_path):
    
#     col1, col2 = st.columns(2)
#     with col1:
#         st.checkbox('Append to existing results', False, key='append_to_results')
#     with col2:
#         if st.button('Reset Results', key='reset_results_button', width='stretch'):
#             st.session_state.show_toast = True
#             st.session_state.toast_text = 'Results Have Been Reset'
#             st.session_state.results = []
#     with st.expander('Load Results'):
#         st.file_uploader(
#             "Upload a csv file for an experiment you already ran",
#             type=['csv'],
#             key="csv_results_uploader",  # A unique key is required for on_change
#             on_change=process_uploaded_results_csv # The callback function
#         )
    
#     if not st.session_state.get('is_running'): 
#         # option to append new results to existing results
    
#         if st.button(
#                 "Start LLM Experiment Run", 
#                 type="primary", 
#                 width='stretch', 
#                 disabled=st.session_state.is_running,
#                 key='start_button_experiment'
#             ):
#             # check required API KEYS
#             correct_keys = True
#             missing_keys=[]
#             if not st.session_state.test:
#                 selected_providers = set([m.split('-')[0] for m in st.session_state.models_to_test])
#                 for provider_name in selected_providers:
#                     api_key = st.session_state['api_keys'][provider_name]
#                     if (not api_key) or (api_key=='abc'):
#                         correct_keys = False
#                         missing_keys.append(provider_name)
        
#             if correct_keys:
#                 st.session_state.is_running = True
#                 st.session_state.stop_requested = False
#                 st.rerun()
#             else:
#                 st.error(f'You need to set your API keys for all providers to test.\n\n Missing keys for {missing_keys}')

#     # The 'Stop' button is only shown while running.
#     if st.session_state.is_running:
#         if st.button(
#                 "STOP Experiment", 
#                 on_click=stop_run_callback,
#                 type="secondary", 
#                 width='stretch',
#                 key='stop_button_experiment'
#             ):
#             st.session_state.is_running = False
#             st.session_state.stop_requested = True
#             st.rerun()

#     if st.session_state.is_running:
#         append_to_results = st.session_state.get('append_to_results')
#         # st.write('append_to_results', append_to_results)
        
#         if not append_to_results:
#             st.info('Creating New Results Dataset')
#             st.session_state['results'] = []
#         else:
#             # st.write('current results', len(st.session_state['results']))
#             st.info('Appending to Existing Results Dataset')
            
#         run_experiments(combinations_to_run)
#         st.session_state.is_running = False
#         st.rerun()
        
        
#     if st.session_state.get('results'):
#         df = create_dataframe_from_results(st.session_state['results'])
#         selected_config_path = st.session_state.get('selected_config_path')
#         show_results(df, selected_config_path)



# def remove_factor_from_state(factor_name, current_round):
#     new_list = [f for f in current_round['factors_list'] if f[0]!=factor_name]
#     st.toast(f'{factor_name} removed')
#     current_round['factors_list'] = new_list

# @st.dialog("Add Factor")
# def show_add_factor_dialog(current_round, round_counter):
#     df = pd.DataFrame({
#         'Level Name': ["Name 1", "Name 2"],
#         'Level Text for LLM': ["Text 1", "Text 2"]
#     })
#     new_factor_name = st.text_input('New Factor Name', key=f'new_factor_name_text_input_{current_round["round_type"]}_{round_counter}')
#     edited_df = st.data_editor(
#     df, 
#     num_rows="dynamic",
#     key=f'data_editor_{current_round["round_type"]}_{round_counter}'
#     )
#     new_factor_names_levels = edited_df['Level Name'].values.tolist()
#     new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
#     new_fact = [
#         new_factor_name, 
#         [{'name':nf_name, 'text':nf_text, 'factor_name':new_factor_name} for nf_name, nf_text in zip(new_factor_names_levels, new_factor_text_levels)]
#     ]
#     if st.button("Add factor", key=f'add_button_{current_round["round_type"]}_{round_counter}'):
#         current_round['factors_list'].append(new_fact)

#         st.session_state.show_toast = True
#         st.session_state.toast_text = f'Factor Added: {new_factor_name}'

#         st.rerun()        

# def show_factor_combinations_mixed(current_round, round_counter):
#     with st.container(border=True):
#         # --- Calculate all values first ---
#         factor_levels = [fl[1] for fl in current_round['factors_list']]
#         # factor_names = [fl[0] for fl in current_round['factors_list']]
#         factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        

#         # st.write(factor_levels)
#         # --- Display the information ---
#         st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
#         l = []
#         for factor_name, factor_level in current_round['factors_list']:
#             # st.write(factor_name)
#             # st.write(factor_level)
#             details = ', '.join([f['name'] for f in factor_level])
#             l.append(f'{factor_name} ({details})')
#         factorial_text_display_details = ' x '.join(l)
#         st.caption(f"{factorial_text_display_details}")
        
# def show_factors_and_levels_mixed(current_round, round_counter):
#     for factor_tuple in (current_round['factors_list']):
#         factor_name = factor_tuple[0] # product
#         factor_levels = factor_tuple[1] # [{"name":'Name 1', "text":'Text 1', "factor_name": "product"},{product|name:'Name 2', product|text:'Text 2'}]
#         # st.write('factor_levels', factor_levels)
#         col_factor, col_remove = st.columns([2,1])
#         with col_factor:
#             st.write(f"`{factor_name}`")
#         with col_remove:
#             if st.button(f'Remove {factor_name}', width='stretch', key=f"remove_button_{current_round['round_type']}_{factor_name}"):
#                 remove_factor_from_state(factor_name, current_round=current_round)
#                 st.rerun()
        
#         original_df = pd.DataFrame(factor_levels)

#         editor_key = f"data_editor_{current_round['round_type']}_{current_round['key']}_{factor_name}"

#         st.data_editor(
#             original_df, 
#             num_rows='dynamic', 
#             key=editor_key,
#             on_change=apply_editor_changes_and_update_state,
#             kwargs={
#                     "factor_name": factor_name,
#                     "state_list_to_update": current_round['factors_list'],
#                     "editor_key": editor_key,
#                     "original_df": original_df,  # Pass the original state to the callback
#                     "has_group_id": False,  # Pass the original state to the callback
#                 },
#             width='stretch',
#             column_order=("name", "text"),
#         )



# def get_num_choices(current_round):
#     # get number of choices
#     num_choices = current_round['choices_shown_in_round']
#     return num_choices


# def mix_block_factor_variables(factors_products, block_variable_products):        
#     full_products = [[*bvp, *fp] for fp in factors_products for bvp in block_variable_products]
#     return full_products
    
# @st.cache_data
# def get_choice_factor_products(factors_list):        
#     factor_levels = [fl[1] for fl in factors_list]
#     factor_products = [list(p) for p in itertools.product(*factor_levels)]
#     return factor_products

# @st.cache_data
# def get_choice_permutations(factors_list, num_choices):
#     if factors_list:
#         factor_products = get_choice_factor_products(factors_list)
#     else:
#         factor_products = []
#     # st.write('factors_list', factors_list)
#     # st.write('factor_products', factor_products)
#     combinations = list(itertools.permutations(factor_products, num_choices))
#     combinations = [list(combo) for combo in combinations]
#     return combinations

# def show_choice_combinations_details_pro(current_round, round_counter):
#     """
#     Displays the experiment setup in a structured and professional card.
#     """
#     with st.container(border=True):
#         # --- Calculate all values first ---
#         factor_levels = [fl[1] for fl in current_round['factors_list']]
#         factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        
#         factor_products = get_choice_factor_products(current_round['factors_list'])
#         # st.write('factor_products', factor_products)
#         total_combinations = len(factor_products)
        
#         num_choices = get_num_choices(current_round)
        
#         # Calculate the number of combinations (nCk)
#         combinations = get_choice_permutations(current_round['factors_list'], num_choices)
#         # st.write('combinations', combinations)
#         total_sets = len(combinations)
        
#         # --- Display the information ---
#         st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
        
#         col1, col2, col3 = st.columns(3)
        
#         with col1:
#             st.metric(
#                 label="Total Factor Combinations", 
#                 value=total_combinations,
#                 help="The total number of unique choices generated by the full factorial design. This is 'n'."
#             )
        
#         with col2:
#             st.metric(
#                 label="Number of Choices Shown", 
#                 value=num_choices,
#                 help="The number of choices the LLM will be asked to evaluate in a single prompt. This is 'r'."
#             )
            
#         with col3:
#             st.metric(
#                 label="Possible Choice Sets", 
#                 value=f"{total_sets:,}", # Adds a comma for thousands
#                 help=f"The total number of unique sets of {num_choices} items that can be created from the {total_combinations} total combinations."
#             )

#         st.caption(f"This is calculated as 'n choose r', or C({total_combinations}, {num_choices}).")

# def show_ranking_combinations_details_pro(current_round, round_counter):

#     with st.container(border=True):
#         # --- Calculate all values first ---
#         rank_permutations_multi_factor = get_rank_permutations_multi_factor(current_round['factors_list'])
#         num_rank_permutations = len(rank_permutations_multi_factor)
#         factor_levels = [fl[1] for fl in current_round['factors_list']]
#         factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.metric(
#                 label="Design", 
#                 value=f"{factorial_text_display} ",
#                 help="Full Factorial Design"
#             )
#             # st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
#         with col2:
#             st.metric(
#                 label="Total Ranking Permutations", 
#                 value=num_rank_permutations,
#                 help="The total number of unique choices generated by the full factorial design. This is 'n'."
#             )            

    
# def show_factor_items_ranking(current_round, round_counter):
#     # st.write(current_round)
#     if not current_round.get('factors_list'):
#         num_items_in_ranking = 0
#         st.info('No factors to rank. Add ***at least*** one factor.')
#         if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
#             show_add_factor_dialog(current_round, round_counter)
#     else:
#         num_items_in_ranking = len(current_round['factors_list'][0][1])
#         st.write(f'{num_items_in_ranking} items to rank')
        
#         show_ranking_combinations_details_pro(current_round, round_counter)
        
#         show_factors_and_levels_mixed(current_round, round_counter)
#         if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
#             show_add_factor_dialog(current_round, round_counter)

# def show_factor_items_choice(current_round, round_counter):
    
#     if not current_round.get('factors_list'):
#         st.info('No choice factors. Add at least one factor.')
        
#         if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
#             show_add_factor_dialog(current_round, round_counter)

#         current_round['factors_list'] = []

#     else:
#         show_choice_combinations_details_pro(current_round, round_counter)


#         st.write('---')
#         show_factors_and_levels_mixed(current_round, round_counter)

#         if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
#             show_add_factor_dialog(current_round, round_counter)
        
#         st.write('---')

#         factor_products = get_choice_factor_products(current_round['factors_list'])
#         max_possible_choices_shown = len(factor_products) if len(factor_products)>2 else 3
#         choices_shown_in_round = st.slider('Number of Choices Shown (*r*)', min_value=2, max_value=max_possible_choices_shown, key=f'slider_{round_counter}_choice', help='''How many choices (r) to show to the LLM in this round. Maximum number of choices depends on n choose r (n: factor combination; r: choices shown to the LLM)''')

#         if current_round['choices_shown_in_round'] != choices_shown_in_round:
#             current_round['choices_shown_in_round'] = choices_shown_in_round
#             st.rerun()
    
# def show_factor_items_scales(current_round, round_counter):
#     if not current_round.get('factors_list'):
#         st.info('No scales factors. Add at least one factor.')
#         current_round['factors_list'] = []

#     else:
#         show_factor_combinations_mixed(current_round, round_counter)
#         st.write('---')
#         show_factors_and_levels_mixed(current_round, round_counter)

#     if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
#         show_add_factor_dialog(current_round, round_counter)


# def show_round_details(current_round, round_counter):
#     '''
#     '''
#     if not current_round: 
#         return

#     st.write(f'## Round {round_counter+1} - {current_round["round_type"].capitalize()}')

#     tab_factor, tab_segment = st.tabs(
#         ['Factors for this Round', 'Segment for this Round'],
#     )
#     with tab_factor:
#         if current_round['round_type']=='ranking':
#             show_factor_items_ranking(current_round, round_counter)
#         elif current_round['round_type']=='choice':
#             show_factor_items_choice(current_round, round_counter)
#         elif current_round['round_type']=='scales':
#             show_factor_items_scales(current_round, round_counter)

#     with tab_segment:
#         show_mixed_segments(current_round, round_counter)

#     # st.write('---')
#     if st.button('# Sample Text Shown to the LLM', width='stretch'):
#         show_sample_mixed_modal(current_round, round_counter)

# @st.dialog("Add New Round")
# def show_add_round_modal():
#     round_type = st.selectbox(
#         'Select Round Type', 
#         [rt.capitalize() for rt in Round_Types.keys()]
#     ).lower()

#     if st.button('Add Round', width='stretch'):
#         new_round_key = str(uuid.uuid4())
#         segments = [
#             {
#                 'segment_text':'',
#                 'segment_id': str(uuid.uuid4())
#             }
#         ]
#         new_round = dict(
#             key=new_round_key,
#             segments=segments,
#             factors_list=[],
#             round_type=round_type
#         )
#         if round_type == 'choice':
#             new_round['choices_shown_in_round'] = 2

#         st.session_state.rounds.append(new_round)
#         st.session_state.selected_round = new_round
        
#         st.rerun()


# def run_experiments(combinations_to_run):
#     api_keys = st.session_state['api_keys']
#     models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
#     total_models_to_test = len(models_objects_to_test)

    
#     total_iterations = total_models_to_test * len(combinations_to_run) * st.session_state.k

#     progress_text = f"Running {total_iterations} experiments..."
#     my_bar = st.progress(0, text=progress_text)
#     progress_tracker = ProgressTracker(counter=0, progress_bar=my_bar, total_iterations=total_iterations)
#     # st.write('current results', len(st.session_state['results']))
    

#     for model in models_objects_to_test:
#         # go through all combinations
#         go_through_all_combinations(
#             combinations_to_run, 
#             model=model, 
#             results=st.session_state['results'],
#             progress_tracker=progress_tracker,
#             )


# def go_through_all_combinations(all_rounds_combinations, model, results, progress_tracker):
#     for i, rounds_combination in enumerate(all_rounds_combinations):
#         # st.write('# combination', i)
#         # st.write('num rounds', len(rounds_combination))

#         # do rounds_combination, k times
#         k = st.session_state['k']
#         run_combination_k_times(rounds_combination, k, model, results, progress_tracker)

# def run_combination_k_times(rounds_combination, k, model, results, progress_tracker):
#     for iteration in range(k):
#         # st.write(f'## Iteration: {iteration}')
#         run_combination(rounds_combination, iteration, model, results, progress_tracker)

# def run_combination(rounds_combination, iteration, model, results, progress_tracker):
#     # start round
#     list_of_messages = []
#     trial_data = {
#             "trial_id": str(uuid.uuid4()),
#             "timestamp": datetime.now().isoformat(),
#             "model_name": model.model_name,
#             "iteration":iteration,
#         }
#     for j, (round_combination, current_round) in enumerate(zip(rounds_combination, st.session_state['rounds'])):
        
#         time.sleep(st.session_state.sleep_amount)

#         # st.write(f'- ### round {j}: {current_round["round_type"]}')
#         # st.write(f'round_combination: ')
#         # st.write(round_combination)

#         if current_round["round_type"]=='ranking':
#             # st.write('round_combination', round_combination)

#             formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, round_combination) 
#             if ranking_display_order:
#                 for ri, rank in enumerate(ranking_display_order):
#                     for key, value in rank.items():
#                         trial_data[f'round{j}_ranking_{ri}_{key}'] = value
#                         # trial_data[f'round{j}_ranking_display_order'] = ranking_display_order

#         elif current_round["round_type"]=='choice':
#             # st.write('round_combination', round_combination)
#             formatted_text, choices_display_order = get_llm_text_mixed_choice(round_combination, current_round)
            
#             # st.write('formatted_text', formatted_text)
#             # st.write('---')
#             # st.write('---')

#             if choices_display_order:
#                 for ci, choice in enumerate(choices_display_order):
#                     for key, value in choice.items():
#                         trial_data[f'round{j}_choice_{ci}_{key}'] = value

#         elif current_round["round_type"]=='scales':
#             formatted_text, factor_display = get_llm_text_mixed_scales(round_combination, current_round)
#             # st.write(factor_display)
#             # st.write('---')
#             # st.write('---')
#             if factor_display:    
#                 for key, value in factor_display.items():        
#                     trial_data[f'round{j}_scales_{key}'] = value
        
#         # st.write(formatted_text)
#         list_of_messages.append({'role':'user','content':formatted_text}) # user_message
        
#         llm_response, parsed_response = call_model_for_user_message(model, '', current_round, history=list_of_messages)
#         # st.write(llm_response.content)
#         # st.write(parsed_response)
#         list_of_messages.append({'role':'assistant', 'content': llm_response.content})

#         trial_data[f'round{j}_{current_round["round_type"]}_llm_response'] = parsed_response if parsed_response else None
#         trial_data[f'round{j}_{current_round["round_type"]}_llm_raw_response'] = llm_response.content

#         if current_round["round_type"]=='ranking':
#             if parsed_response:
#                 try:
#                     trial_data[f'round{j}_ranking_llm'] = [ranking_display_order[response-1] for response in parsed_response]
#                 except Exception as e:
#                     trial_data[f'round{j}_ranking_llm'] = e
#         elif current_round["round_type"]=='choice':
#             if parsed_response:
#                 try:
#                     trial_data[f'round{j}_choice_llm'] = choices_display_order[parsed_response-1]
#                 except Exception as e:
#                     trial_data[f'round{j}_choice_llm'] = e
#     results.append(trial_data)

#     progress_tracker.counter+=1
#     progress_made = int((progress_tracker.counter/progress_tracker.total_iterations)*100)
#     progress_tracker.progress_bar.progress(progress_made, text=f"Experiments executed: {progress_tracker.counter}")
#     # end round

# def call_model_for_user_message(model, user_message, current_round, history=[]):
#     if st.session_state.test:
#         if current_round['round_type']=='ranking':
#             permutations = get_rank_permutations_multi_factor(current_round)
#             num_to_rank = len(permutations[0])
#             # st.write('num_to_rank', num_to_rank)
#             llm_ranks = list(range(1, num_to_rank+1))
#             random.shuffle(llm_ranks)
#             llm_ranks_json = json.dumps(llm_ranks)

#             response_data = LLMResponse(
#                 content=llm_ranks_json,
#                 response_time_ms=1,
#                 raw_response='raw_response'
#             )
#             parsed_choice = json.loads(llm_ranks_json)  

#         elif current_round['round_type']=='choice':
#             num_choices = current_round['choices_shown_in_round']
#             choices_list = list(range(1, num_choices+1))
#             llm_choice = random.choice(choices_list)
#             llm_choice_json = json.dumps(llm_choice)

#             response_data = LLMResponse(
#                 content=llm_choice_json,
#                 response_time_ms=1,
#                 raw_response='raw_response'
#             )
#             parsed_choice = json.loads(llm_choice_json)

#         elif current_round['round_type']=='scales':
#             llm_scales = random.choice(range(1, 8))
#             llm_scales_json = json.dumps(llm_scales)
            
#             response_data = LLMResponse(
#                 content=llm_scales_json,
#                 response_time_ms=1,
#                 raw_response='raw_response'
#             )
#             parsed_choice = json.loads(llm_scales_json)
#     else: 
#         response_data = model.query( 
#             st.session_state.system_prompt, 
#             user_message, 
#             conversation_history=history
#         )
    
#         if 'error' in str(response_data.raw_response).lower():
#             parsed_choice = None
#         else:
#             try:
#                 parsed_choice = json.loads(response_data.content.strip())
#             except Exception as e:
#                 parsed_choice = e
#     return response_data, parsed_choice




# def filter_factors_list(factors_list, round_segment_variables):
#     return [fac for fac in (factors_list) if fac[0] in round_segment_variables]

# def group_id_in_segment_variables(round_segment_variables, block_variable_name, block_variable_level, block_variables):
#     group_id = block_variable_level.get('group_id') or block_variable_level.get('name')
    
#     alias_levels = []
#     for bvar_name, bvar_levels in block_variables:
#         for bvar_level in bvar_levels:
#             if (((bvar_level.get('group_id')==group_id) or (bvar_level.get('name')==group_id)) and (bvar_name in round_segment_variables)):
#                 alias_levels.append(bvar_level)
#     return alias_levels

# def remove_dupes_from_complex_list(complex_list):
#     # st.write('len(complex_list)', len(complex_list))
#     new_list = []
#     seen = set()

#     for item in complex_list:
#         key = json.dumps(item, sort_keys=True)
#         if key not in seen:
#             new_list.append(item)
#             seen.add(key)
            
#     # st.write('len(new_list)', len(new_list))
    
#     return new_list

# @st.cache_data
# def get_rounds_factor_combinations(rounds):
#     rounds_factor_combinations = []
#     for round_counter, current_round in enumerate(rounds):
#         # st.write(round_counter)
        
#         round_segment_variables = get_segment_text_from_segments_list(current_round)
#         # st.write('round_segment_variables', round_segment_variables)

#         filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
#         # st.write(f'round {round_counter}','filtered_factors_list', filtered_factors_list)
        
#         if current_round['round_type'] == 'ranking':
#             factor_levels_rank_permutations = get_rank_permutations_multi_factor(filtered_factors_list)
#             # st.write(factor_levels_rank_permutations)
#             rounds_factor_combinations.append(factor_levels_rank_permutations)
        
#         elif current_round['round_type'] == 'choice':
#             num_choices = get_num_choices(current_round)
#             # st.write('filtered_factors_list', filtered_factors_list)
#             combinations = get_choice_permutations(filtered_factors_list, num_choices)
#             # st.write('combinations', combinations)
#             rounds_factor_combinations.append(combinations)
            
#         elif current_round['round_type'] == 'scales':
#             factor_products = get_choice_factor_products(filtered_factors_list)
#             rounds_factor_combinations.append(factor_products)
        
#     return rounds_factor_combinations

# @st.cache_data
# def get_all_rounds_combinations(rounds_factor_combinations):
#     all_rounds_combinations = list(itertools.product(*rounds_factor_combinations))
#     all_rounds_combinations = [list(combo) for combo in all_rounds_combinations]
#     return all_rounds_combinations

# @st.cache_data
# def get_block_added_all_rounds_combinations_deduped(all_rounds_combinations, rounds, block_variables):

#     if len(block_variables)<1:
#         return all_rounds_combinations
#     else:
#         # use one factor (each factor is a different representation of the same construct)
#         block_variable = block_variables[0]
#         block_variable_name, block_variable_levels = block_variable
        
#     # st.write('block_variables', block_variables)
#     # st.write('all_rounds_combinations', all_rounds_combinations[:10])
#     # return
#     # st.write('len(all_rounds_combinations)', len(all_rounds_combinations))
    
#     new_all_rounds_combinations = []
#     for i, round_combination in enumerate(all_rounds_combinations):
#         # st.write(f'combination {i}')
#         for b, block_variable_level in enumerate(block_variable_levels): 
#             # insert each block_variable_level in the round_combination
#             # same block_variable_level for all rounds
#             new_round_combination = []
#             for s, (single_round_combination, current_round) in enumerate(zip(round_combination, rounds)):
#                 # st.write(f'single_round_combination {s}')
#                 round_segment_variables = get_segment_text_from_segments_list(current_round)

#                 alias_levels = group_id_in_segment_variables(round_segment_variables, block_variable_name, block_variable_level, block_variables)
#                 # st.write(f"### {current_round['round_type']}")
#                 # st.write('round_segment_variables', round_segment_variables)
#                 # st.write('block_variable_name', block_variable_name)
#                 # st.write('alias_levels', alias_levels)
#                 # if alias_levels:
#                 #     new_combination = [alias_levels, *combination]
                                
#                 if current_round['round_type']=='ranking': 
#                     # list of lists (levels to rank)
#                     # st.write('block_variable_name', block_variable_name)
#                     # st.write('block_variable_level', block_variable_level)
#                     # st.write('current_round', f"\n\n### **{current_round['round_type']}**")
#                     # st.write(f'adding {block_variable_level["country|name"]}')
#                     if alias_levels:
#                         new_single_round_combination = [alias_levels, *single_round_combination]
#                     else:
#                         new_single_round_combination = single_round_combination[:]
#                     # st.write('single_round_combination', single_round_combination)
                     
#                     # return
#                 elif current_round['round_type']=='scales': 
#                     # list of dict (factor levels)
#                     if alias_levels:
#                         new_single_round_combination = [*alias_levels, *single_round_combination]
#                     else:
#                         new_single_round_combination = single_round_combination[:]

#                 elif current_round['round_type']=='choice': 
#                     # list of lists (choices_shown_in_round items)
#                     if alias_levels:
#                         new_single_round_combination = [alias_levels, *single_round_combination]
#                     else:
#                         new_single_round_combination = single_round_combination[:]

#                 new_round_combination.append(new_single_round_combination)
#             new_all_rounds_combinations.append(new_round_combination)
#         # st.write('---')
#     # st.write('len(new_all_rounds_combinations)', len(new_all_rounds_combinations))
#     # st.write('new_all_rounds_combinations[:2]', new_all_rounds_combinations[:2])

#     block_added_all_rounds_combinations_deduped = remove_dupes_from_complex_list(new_all_rounds_combinations)

#     return block_added_all_rounds_combinations_deduped



# def show_sample_ranking(current_round, round_counter):
#     rounds = st.session_state.get('rounds')
#     if not rounds: 
#         st.warning('No Rounds. Add at least one round.')
#         return

#     rounds_factor_combinations = get_rounds_factor_combinations(rounds)
    
#     all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)    
#     # st.write(all_rounds_combinations)
#     block_variables = st.session_state.get('block_variables', [])   
#     # st.write('block_variables', block_variables) 
#     block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations, rounds, block_variables)
#     # st.write(block_added_all_rounds_combinations_deduped[:2])
#     # get a combination
#     combo = block_added_all_rounds_combinations_deduped[0]
#     # get current round from combination
#     current_round_combo = combo[round_counter]
    
    
#     if st.button('Refresh Sample', width='stretch' ,key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
#         combo = random.choice(block_added_all_rounds_combinations_deduped)
#         # get current round from combination
#         current_round_combo = combo[round_counter]
    
            
#     formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, current_round_combo) 
    
#     if not ranking_display_order:
#         st.warning('No Ranking Factors')
#     else:
#         st.write('### Display Order')
#         for i, rank_display in enumerate(ranking_display_order):
#             st.write(f"{i+1}. {rank_display}" )
#         st.write('---')
    
#     st.write('### LLM Text')
#     st.write(formatted_text) 
#     st.write('---')
    
# def show_sample_choice(current_round, round_counter):
#     rounds = st.session_state.get('rounds')
#     if not rounds: 
#         st.warning('No Rounds. Add at least one round.')
#         return

#     rounds_factor_combinations = get_rounds_factor_combinations(rounds)
    
#     # st.write('rounds_factor_combinations', rounds_factor_combinations)
    
#     all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)
        
#     # st.write('all_rounds_combinations', all_rounds_combinations)
    
    
#     block_variables = st.session_state.get('block_variables', [])    
#     block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations, rounds, block_variables)
    
#     # st.write('block_added_all_rounds_combinations_deduped', block_added_all_rounds_combinations_deduped)
    
    
#     # get a combination
#     combo = block_added_all_rounds_combinations_deduped[0]
#     # st.write('combo', combo)
#     # get current round from combination
#     current_round_combo = combo[round_counter]
    
#     # st.write([comb[round_counter] for comb in block_added_all_rounds_combinations_deduped])    

#     if st.button('Refresh Sample', width='stretch' , key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
#         combo = random.choice(block_added_all_rounds_combinations_deduped)
#         # get current round from combination
#         current_round_combo = combo[round_counter]
    

#     formatted_text, choices_display_order = get_llm_text_mixed_choice(current_round_combo, current_round) 

#     if not choices_display_order:
#         st.warning('No Choice Factors.')
#     else:
#         st.write('### Display Order')
#         for i, choice_display in enumerate(choices_display_order):
#             st.write(f"{i+1}. {choice_display}" )
#         st.write('---')

#     st.write('### LLM Text')
#     st.write(formatted_text)
#     st.write('---')

# def show_sample_scales(current_round, round_counter):
#     rounds = st.session_state.get('rounds')
#     if not rounds: 
#         st.warning('No Rounds. Add at least one round.')
#         return

#     rounds_factor_combinations = get_rounds_factor_combinations(rounds)
#     # st.write('rounds_factor_combinations', rounds_factor_combinations)
#     all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)    
#     # st.write('all_rounds_combinations', all_rounds_combinations)

#     block_variables = st.session_state.get('block_variables', [])    
#     block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations, rounds, block_variables)
#     # st.write('block_added_all_rounds_combinations_deduped', block_added_all_rounds_combinations_deduped)
    
#     # get a combination
#     combo = block_added_all_rounds_combinations_deduped[0]
#     # st.write('combo', combo)
#     # get current round from combination
#     current_round_combo = combo[round_counter]
    
#     # st.write(current_round_combo )
    
#     if st.button('Refresh Sample', width='stretch' , key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
#         combo = random.choice(block_added_all_rounds_combinations_deduped)
#         current_round_combo = combo[round_counter]
        
#     # st.write('current_round_combo', current_round_combo)
#     formatted_text, factors_display = get_llm_text_mixed_scales(current_round_combo, current_round) 
#     if factors_display:
#         st.write('### Factor Combination: ')
#         st.write(f'{factors_display}')
#         st.write('---')
#     else:
#         st.write('No **Factor** or **Block Variable** in Segment')
#         st.write('---')
        
#     st.write('### LLM Text')
#     st.write(formatted_text)
#     st.write('---')


# def show_experiment_combinations():
#     models_to_test = st.session_state.models_to_test
#     total_models_to_test = len(models_to_test)

#     rounds = st.session_state.get('rounds')
#     if not rounds: 
#         return

#     rounds_factor_combinations = get_rounds_factor_combinations(rounds)
#     # st.write('rounds_factor_combinations', rounds_factor_combinations)
#     all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)
            
#     block_variables = st.session_state.get('block_variables', [])    
#     if block_variables:
#         block_var_name, block_var_levels = block_variables[0]
#         # st.write(block_var_levels)
#         num_block_variable_levels = len(block_var_levels)        
#     else:
#         num_block_variable_levels = 0    
#     block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations, rounds, block_variables)
    
 
#     total_combinations = len(block_added_all_rounds_combinations_deduped)
 
#     k_value = st.session_state.k
#     # Use a container to group the experiment summary
#     with st.container(border=True):
#         st.subheader("Experiment Configuration Summary")
        
#         # Use a divider to separate parameters from metrics
#         st.divider()

#         # Use columns for a dashboard-like layout of the key metrics
#         col1, col2, col3, col4, col5 = st.columns(5)
#         with col4:
#             st.metric(label="Total Combinations", value=len(block_added_all_rounds_combinations_deduped))
#         with col2:
#             st.metric(label="***k*** Value", value=k_value)
#         with col1:
#             st.metric(label="Total Models", value=total_models_to_test)
#         with col3:
#             st.metric(label="Block Variable Levels", value=num_block_variable_levels)


#         with st.expander('Choose a Subset of Combinations to Run (optional)'):
#             combinations_to_run = choose_a_subset_of_combinations(block_added_all_rounds_combinations_deduped)
        
#         # 3. Show a preview and the final run button
#         # st.markdown("---")
#         st.info(f"***Executing {len(combinations_to_run)} out of {total_combinations} combinations***")
#         # st.markdown("---")
        
#         total_iterations = total_models_to_test * len(combinations_to_run) * st.session_state.k
#         st.metric(label="Total Iterations = Total Models × *k* × Combinations to Run", value=f"{total_iterations}")
#         # st.write(f"### Total Iterations\n\nTotal Models × *k* × Total Combinations = {total_iterations}")
#     return combinations_to_run

# def show_round_container():
#     st.write('### User Message Rounds')
#     if not st.session_state.get('selected_round'):
#         if not st.session_state.get('rounds'):
#             st.session_state['selected_round'] = None
#             st.session_state['selected_round_counter'] = None
#         else:
#             st.session_state['selected_round'] = st.session_state.rounds[0]
#             st.session_state['selected_round_counter'] = 0

#     with st.container(border=True):
#         if not st.session_state.get('rounds'):
#             st.info('**No Rounds:** Create at least one round of conversation for the LLM.')
#             if st.button(":material/add: Add Round", width='stretch', type="secondary", key='add_round_modal_button_empty_round'):
#                 show_add_round_modal()

#         else:
#             col_master, col_details = st.columns(
#                 [1,3], 
#                 gap='medium',
#                 vertical_alignment='top',
#                 border=False
#             )
#             with col_master:
#                 st.write('## Rounds')
#                 row = st.columns([2, 1], gap='medium') # Create a small column for the delete button
#                 for round_counter, current_round in enumerate(st.session_state.rounds):
#                     with row[0]: # round buttons
#                         # Check if this is the selected round
#                         is_selected = (current_round['key'] == st.session_state.selected_round['key'])
                        
#                         # Set the button type accordingly
#                         button_type = "primary" if is_selected else "secondary"

#                         if st.button(
#                             f'Round {round_counter+1} - {current_round["round_type"].capitalize()}',
#                             key=f"round_btn_{current_round['key']}",
#                             type=button_type,
#                             width='stretch'
#                         ):
#                             st.session_state.selected_round = current_round
#                             # Rerun to ensure the button style updates immediately
#                             st.rerun() 
                        
#                         if is_selected:
#                             st.session_state['selected_round_counter'] = round_counter

#                     with row[1]: # remove round
#                         if st.button("✕", key=f"del_{current_round['key']}", help="Delete this round"):
#                             removed_round = st.session_state.rounds.pop(round_counter)

#                             if not st.session_state.rounds:
#                                 # Handle case where all rounds are deleted
#                                 st.session_state['selected_round'] = None
#                                 st.session_state['selected_round_counter'] = None
#                                 pass 
#                             else:
#                                 st.session_state.selected_round = st.session_state.rounds[0]

#                             st.session_state.show_toast = True
#                             st.session_state.toast_text = f'Round {round_counter+1} - {current_round["round_type"].capitalize()} deleted'

#                             st.rerun()

#                 with row[0]:                                
#                     if st.button(":material/add: Add New Round", width='stretch', type="secondary", key='add_round_modal_button', help='Add New Round'):
#                         show_add_round_modal()

#             with col_details:
#                 with st.container(border=False):
#                     show_round_details(st.session_state.get('selected_round'), st.session_state.get('selected_round_counter', 0))

# def render_mixed_experiment(selected_config_path):
    
#     show_toast()
#     st.header("Run a Behavioral Experiment with an LLM as the Participant")
#     st.caption("Select a configuration file, choose the LLMs, and modify the run parameters.")
    
#     st.file_uploader(
#         "*Upload a YAML configuration file for a predefined experiment.*",
#         type=['yaml', 'yml'],
#         key="yaml_uploader",  # A unique key is required for on_change
#         on_change=process_uploaded_yaml # The callback function
#     )


#     with st.container(border=True, horizontal=True):
#         col_config, col_reset = st.columns(2)
#         with col_config:
#             show_experiment_configs_selectors()
#             show_block_variables()


#         with col_reset:
#             with st.expander('System Prompt'):
#                 st.text_area(
#                     label='System Prompt', 
#                     value=st.session_state.get('system_prompt', ''), 
#                     placeholder='''Type in your System Prompt''',
#                     key='system_prompt' 
#                 )
#             if st.button('Reset Configuration', width='stretch'):
#                 reset_config()
            

#     show_round_container()

#     config_to_save = get_config_to_save(factors_to_save)
#     # st.write(config_to_save)

#     combinations_to_run = show_experiment_combinations()

#     show_download_save_config(config_to_save, selected_config_path)

#     show_mixed_experiment_execution(combinations_to_run, selected_config_path)

 
        
# def load_experiment_config(path_str):
#     """Loads a YAML config file content from a given file path."""
#     try:
#         config_path = Path(path_str)
#         with open(config_path, 'r', encoding='utf-8') as f:
#             # Use yaml.safe_load for security
#             config_dict = yaml.safe_load(f)
#         return config_dict
#     except FileNotFoundError:
#         st.error(f"Configuration file not found at: {path_str}")
#         return None
#     except yaml.YAMLError as e:
#         st.error(f"Error parsing YAML file: {e}")
#         return None

# def show_toast():
#     if st.session_state.get('show_toast'):
#         st.toast(st.session_state.get('toast_text'))
#         st.session_state.show_toast = False


# def show_download_save_config(config_to_save, selected_config_path):
#     if not is_prod:
#         col_download, col_save = st.columns(2)
#         with col_download:
#             st.download_button(
#             "Download Config",
#             yaml.dump(config_to_save, sort_keys=False, Dumper=NoAliasDumper),
#             f"config.yaml",
#             "text/yaml",
#             key='download-yaml',
#             width='stretch'
#             )
        
#         with col_save:
#             if st.button("Save config", type="secondary", width='stretch'):
#                 st.write(selected_config_path)
#                 with open(selected_config_path,'w') as f:
#                     yaml.dump(config_to_save, f, sort_keys=False, Dumper=NoAliasDumper)
#     else:
#         st.download_button(
#             "Download Config",
#             yaml.dump(config_to_save, sort_keys=False, Dumper=NoAliasDumper),
#             f"{selected_config_path}",
#             "text/yaml",
#             key='download-yaml',
#             width='stretch'
#         )
        
# def start_run_callback():
#     st.session_state.is_running = True
#     st.session_state.stop_requested = False

# def stop_run_callback():
#     st.session_state.stop_requested = True
#     # st.toast("Experiment interruption requested. Waiting for current iteration to finish.")
 
 
    



# @dataclass
# class ProgressTracker:
#     """Standardized response object for all LLM queries."""
#     counter: int
#     progress_bar: Any
#     total_iterations: int


# def Experiment():
#     config_path = config_paths['mixed']
#     config = load_experiment_config(config_path)

#     # Store the current config path in session_state if needed later
#     st.session_state.selected_config_path = config_path
        
#     # On first render, Reset all relevant state variables from the new config
#     for factor_to_load in config.keys():
#         if st.session_state.get(factor_to_load) is None:
#             # st.write('setting', factor_to_load)
#             st.session_state[factor_to_load] = config.get(factor_to_load)

#     # st.write(st.session_state)
#     # Render the page
#     render_mixed_experiment(selected_config_path=config_paths['mixed'])



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