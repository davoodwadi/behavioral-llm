import streamlit as st
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

from experiment_choice import StExperimentRunnerChoice, StExperimentRunnerRanking, StExperimentRunnerScales
from experiment_runners import StExperimentRunnerMixed
import itertools
from utils import get_llm_text_choice, get_llm_text_rank, get_llm_text_scales, get_llm_text_mixed_rank, get_llm_text_mixed_choice, get_llm_text_mixed_scales
from models import get_model, LLMResponse

class NoAliasDumper(yaml.Dumper):
    def ignore_aliases(self, data):
        return True
# --- Configuration (Runs once at the top) ---
st.set_page_config(
    page_title="Behavioral LLM Experiment Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

APP_ENVIRONMENT = os.environ.get('APP_ENV', 'dev') 

# is_prod = "SPACE_ID" in os.environ
is_prod = (APP_ENVIRONMENT == 'production')
# is_prod = True
# Initialize session state flags
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

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
    "deepinfra-deepseek-ai/DeepSeek-V3.1-Terminus",
    "deepinfra-meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "deepinfra-meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "deepinfra-moonshotai/Kimi-K2-Instruct-0905",
]
if not is_prod:
    ALL_MODELS.append('test-test')

st.session_state['api_keys'] = {m.split('-')[0]:None for m in ALL_MODELS}

Round_Types = {
    'choice': {
        "Segment_Types": ['Fixed Segment', 'Choice Segment']
    },
    'ranking': {
        "Segment_Types": ['Fixed Segment', 'Ranking Segment']
    },
    'scales': {
        "Segment_Types": ['Fixed Segment', 'Treatment Segment']
    }
}
        
factors_to_save_dict = dict()
factors_to_save_dict['mixed'] = [
    'system_prompt',
    'rounds',
    'block_variables',
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'randomize',
    'paper_url',
    'api_keys'
]

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'thread' not in st.session_state:
    st.session_state.thread = None

config_paths = {
    'choice': 'config/choice/COO/country of origin choice.yaml',
    'scales': 'config/scales/ad_framing_and_scarcity.yaml',
    'ranking': 'config/rank/Balabanis 2004/Domestic Country Bias.yaml',
    'mixed': 'config/mixed/Domestic Country Bias.yaml',
}

def process_uploaded_yaml():
    """
    This function is called ONLY when the file_uploader's state changes.
    It reads the uploaded file and updates the session state.
    """
    # Get the uploaded file from session state using the key.
    uploaded_file = st.session_state.yaml_uploader
    
    if uploaded_file is not None:
        try:
            # To read the file as a string
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            string_data = stringio.read()
            yaml_data = yaml.safe_load(string_data)
            # st.write(yaml_data)
            # store the config in state for reset_config
            st.session_state['yaml_data'] = yaml_data
            # Update session state with the keys from the YAML file
            for state_var_name in yaml_data.keys():
                st.session_state[state_var_name] = yaml_data.get(state_var_name)
                # st.write(state_var_name, yaml_data.get(state_var_name))
                # st.write(state_var_name)
            
            # reset selected_round and selected_round_counter
            if yaml_data.get('rounds'):
                st.session_state.selected_round = st.session_state.rounds[0]
                st.session_state.selected_round_counter = 0

            if yaml_data.get('api_keys'):
                api_keys = yaml_data.get('api_keys')
                for provider_name, api_key in api_keys.items():
                    st.session_state[f'{provider_name}_api_key'] = api_key

            st.success(f"Successfully loaded configuration from '{uploaded_file.name}'!")
        
        except Exception as e:
            st.error(f"Error processing YAML file: {e}")

def show_experiment_configs_selectors():
    with st.expander('Experiment Configs'):
        k = st.session_state.get('k', 1)
        # st.write(k)
        st.number_input(
            "Number of Iterations", value=k, placeholder="Type a number...",
            key = 'k'
        )

        test = st.session_state.get('test', True)
        st.toggle('Run Mock Experiments', value=test, key='test')
        
        sleep_amount = st.session_state.get('sleep_amount', 0.1)
        st.number_input(
            'Amount to pause between LLM API calls (seconds)', 
            value=sleep_amount,
            key = 'sleep_amount'
        )

        models_to_test = st.session_state.get('models_to_test', [])
        filtered_models_to_test = [m for m in models_to_test if m in ALL_MODELS]
        if not filtered_models_to_test:
            filtered_models_to_test = ALL_MODELS[0]
        st.multiselect(
            'LLMs to Test', 
            ALL_MODELS, 
            default=filtered_models_to_test,
            key='models_to_test'
        )

        # Based on st.session_state.models_to_test
        # get required api keys
        if st.session_state.get('api_keys'):
            api_keys = st.session_state.get('api_keys')
        else:
            api_keys = dict()

        if not st.session_state.test:
            selected_providers = set([m.split('-')[0] for m in st.session_state.models_to_test])
            for provider_name in selected_providers:
                the_key = api_keys.get(provider_name, '')

                st.text_input(
                    f'{provider_name.capitalize()} API key',
                    value=the_key,
                    type="password", 
                    key=f'{provider_name}_api_key'
                )

                the_widget_key = st.session_state.get(f'{provider_name}_api_key')
                st.session_state['api_keys'][provider_name] = the_widget_key
                        
        # 

        paper_url = st.session_state.get('paper_url', '')
        st.text_input(
            "Paper URL (optional)", 
            value=paper_url,
            key='paper_url'
        )

        randomize = st.session_state.get('randomize', True)
        st.checkbox(
            'Randomize Items Displayed in Ranking/Choice Segments', 
            randomize,
            key='randomize'
        )

def reset_config():
    if st.session_state.get('yaml_data'):
        config = st.session_state.get('yaml_data')
    else:
        config = load_experiment_config(st.session_state.selected_config_path)
    
    # Reset all relevant state variables from the new config
    for factor_to_load in config.keys():
        st.session_state[factor_to_load] = config.get(factor_to_load)
    # st.write(config['k'])
    # reset selected_round and selected_round_counter
    st.session_state.selected_round = st.session_state.rounds[0]
    st.session_state.selected_round_counter = 0

    if config.get('api_keys'):
        api_keys = config.get('api_keys')
        for provider_name, api_key in api_keys.items():
            st.session_state[f'{provider_name}_api_key'] = api_key

    st.rerun()
    

@st.dialog("Add Block Variable")
def show_add_block_variable_dialog():
    df = pd.DataFrame({
        'Level Name': ["Name 1", "Name 2"],
        'Level Text for LLM': ["Text 1", "Text 2"],
        'Group ID': ["", ""],
    })
    new_factor_name = st.text_input('New Block Variable Name', key=f'new_block_variable_name_text_input')
    edited_df = st.data_editor(
    df, 
    num_rows="dynamic",
    key=f'data_editor_block_variable'
    )
    new_factor_names_levels = edited_df['Level Name'].values.tolist()
    new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
    new_factor_group_id = edited_df['Group ID'].values.tolist()
    # st.write(new_factor_group_id)
    new_fact = [new_factor_name, [{f'{new_factor_name}|name':nf_name, f'{new_factor_name}|text':nf_text, f'{new_factor_name}|group_id':nf_group_id} for nf_name, nf_text, nf_group_id in zip(new_factor_names_levels, new_factor_text_levels, new_factor_group_id)]]
    if st.button("Add Block Variable", key=f'add_button_block_variable'):
        st.session_state['block_variables'].append(new_fact)

        st.session_state.show_toast = True
        st.session_state.toast_text = f'Block Variable Added: {new_factor_name}'

        st.rerun() 

def remove_block_variable(variable_name):
    if st.session_state.get('block_variables'):
        st.session_state['block_variables'] = [bvar for bvar in st.session_state['block_variables'] if bvar[0]!=variable_name]
        st.toast(f'Block Variable: {variable_name} Removed.' )
    # st.rerun()

def show_block_variables():
    with st.expander('Block Variables (optional)'):
        st.info('Only one block variable is allowed here. Add more variables to indicate different variations of the same variable (e.g., country and nationality). For multiple variables use `group_id` to connect levels to each other.')
        if st.session_state.get('block_variables'):
            for factor_tuple in (st.session_state['block_variables']):
                factor_name = factor_tuple[0] # product
                factor_levels = factor_tuple[1] # [{product|name:'Name 1', product|text:'Text 1'},{product|name:'Name 2', product|text:'Text 2'}]
                # st.write('something')
                col_factor, col_remove = st.columns([2,1])
                with col_factor:
                    st.write(f'***{factor_name}***')
                with col_remove:
                    if st.button(f'Remove {factor_name}', width='stretch', key=f"remove_button_block_variables_{factor_name}"):
                        remove_block_variable(factor_name)
                        st.rerun()
                
                factor_levels_display = []
                for level_dict in factor_levels:
                    clean_dict = {k.split('|')[1]: v for k, v in level_dict.items()}
                    # st.write(clean_dict)
                    factor_levels_display.append(clean_dict)

                original_df = pd.DataFrame(factor_levels_display)
                # st.dataframe(original_df)
                
                editor_key = f"data_editor_block_variables_{factor_name}"

                st.data_editor(
                    original_df, 
                    num_rows='dynamic', 
                    key=editor_key,
                    on_change=apply_editor_changes_and_update_state,
                    kwargs={
                            "factor_name": factor_name,
                            "state_list_to_update": st.session_state['block_variables'],
                            "editor_key": editor_key,
                            "original_df": original_df,  # Pass the original state to the callback
                            "has_group_id": True,  # Pass the original state to the callback
                        },
                    width='stretch',
                    column_order=("name", "text", 'group_id'),
                )

        else:
            st.session_state.block_variables = []
            st.info('No Block Variables. You can optionally add them here.')

        if st.button(f'Add Block Variable'):
            show_add_block_variable_dialog()

def apply_editor_changes_and_update_state(factor_name, editor_key, original_df, state_list_to_update, has_group_id=False):
    """
    Callback to apply the delta from st.data_editor to the original DataFrame
    and then update the main application state.
    This function is robust against empty/NaN rows.
    """
    # 1. Get the delta object from session_state. This is the source of truth for changes.
    try:
        delta = st.session_state[editor_key]
    except KeyError:
        # This can happen in rare edge cases, so we exit gracefully.
        return

    # Start with a clean copy of the data that was originally passed to the editor
    final_df = original_df.copy()

    # 2. Apply DELETED rows first
    # delta["deleted_rows"] is a list of original integer indices to delete
    if delta["deleted_rows"]:
        # The `errors='ignore'` flag prevents crashes if an index is already gone
        final_df = final_df.drop(index=delta["deleted_rows"], errors='ignore')

    # 3. Apply ADDED rows
    # delta["added_rows"] is a list of new row dictionaries
    if delta["added_rows"]:
        added_df = pd.DataFrame(delta["added_rows"])
        final_df = pd.concat([final_df, added_df], ignore_index=True)
        
    # 4. Apply EDITED rows last
    # delta["edited_rows"] is a dict like {row_index: {col_name: new_value}}
    for row_index_str, changed_data in delta["edited_rows"].items():
        row_index = int(row_index_str)
        for col_name, new_value in changed_data.items():
            # Use .at for fast, label-based single-cell assignment
            final_df.at[row_index, col_name] = new_value

    # 5. --- CRITICAL DATA CLEANING STEP ---
    final_df.fillna('', inplace=True)

    final_df.reset_index(drop=True, inplace=True)

    # 6. Convert the clean, final DataFrame back into your required state format
    final_records = final_df.to_dict(orient='records')
    
    if has_group_id:
        formatted_levels = [
            {f"{factor_name}|name": level.get('name', ''), f"{factor_name}|text": level.get('text', ''), f"{factor_name}|group_id": level.get('group_id', '')} for level in final_records
        ]
    else:
        formatted_levels = [
            {f"{factor_name}|name": level.get('name', ''), f"{factor_name}|text": level.get('text', '')}
            for level in final_records
        ]
        
    new_factor_data = [factor_name, formatted_levels]

    # 7. Find and update the factor in the main state object
    try:
        index = [i for i, fac in enumerate(state_list_to_update) if fac[0] == factor_name][0]
        state_list_to_update[index] = new_factor_data
    except IndexError:
        st.error(f"Could not find factor '{factor_name}' to update.")


    st.rerun()

def show_mixed_segments(current_round, round_counter):
    st.info('Use `{factor_name}` to add placeholders for factors\n\nExample: {product} or {price}')

    with st.container(border=True):
        
        factor_names_raw = [f[0] for f in current_round['factors_list']]
        block_variables = st.session_state.get('block_variables')
        
        if block_variables:
            factor_names_raw.extend([f[0] for f in block_variables])
        
        factor_names_display = ['`{' + f + '}`' for f in factor_names_raw]
        factor_names_str = ' '.join(factor_names_display)
        
        segment = current_round.get('segment')
        text_content  = st.text_area(
            f"Round Segment\n\nFactors to use: {factor_names_str}", 
            value=segment, 
            key=f"segment_text_{current_round['key']}"
        )
        current_round['segment'] = text_content
        # st.write('current_round[segment]', current_round['segment'])
        # 3. Check for missing factors
        missing_factors = []
        # Loop through the raw factor names ('Sales', 'Marketing', etc.)
        if current_round['segment']:
            for factor in factor_names_raw:
                # st.write(factor)
                # Create the placeholder string to search for ('{Sales}', '{Marketing}', etc.)
                placeholder = f"{{{factor}}}"
                if placeholder not in current_round['segment']:
                    missing_factors.append(factor)
        else:
            missing_factors = factor_names_raw
        # st.write('missing_factors', missing_factors)
        # if text_content and missing_factors:
        #     # Format the missing factors for a clear warning message
        #     missing_factors_formatted = ", ".join([f"`{{{f}}}`" for f in missing_factors])
        #     st.warning(f"**Missing or incorrectly formatted factors:** {missing_factors_formatted}")
    # st.write("current_round['segment']", current_round['segment'])


def get_config_to_save():
    factors_to_save = factors_to_save_dict['mixed']
    # config = load_experiment_config(st.session_state.selected_config_path)
    config_to_save = dict()
    for factor_to_save in factors_to_save:
        config_to_save[factor_to_save] = st.session_state.get(factor_to_save)
    return config_to_save

def get_rank_permutations(current_round):
    factors_list = current_round.get('factors_list')
    if not factors_list:
        return []
    factor_name, factor_levels_rank = factors_list[0]
    factor_levels_rank_permutations = list(itertools.permutations(factor_levels_rank, len(factor_levels_rank)))
    factor_levels_rank_permutations_list = [list(permutation) for permutation in factor_levels_rank_permutations]
    return factor_levels_rank_permutations_list

def get_rank_permutations_multi_factor(factors_list):
    if not factors_list:
        return []

    factor_products = get_choice_factor_products(factors_list)
    factor_levels_rank_permutations = list(itertools.permutations(factor_products, len(factor_products)))
    factor_levels_rank_permutations_list = [list(permutation) for permutation in factor_levels_rank_permutations]
    return factor_levels_rank_permutations_list



@st.dialog('Sample User Message Shown to the LLM')
def show_sample_mixed_modal(current_round, round_counter):
    if current_round['round_type']=='ranking':
        show_sample_ranking(current_round, round_counter)
    elif current_round['round_type']=='choice':
        show_sample_choice(current_round, round_counter)
    elif current_round['round_type']=='scales':
        show_sample_scales(current_round, round_counter)

def show_download_save_config(config_to_save, selected_config_path):
    if not is_prod:
        col_download, col_save = st.columns(2)
        with col_download:
            st.download_button(
            "Download Config",
            yaml.dump(config_to_save, sort_keys=False, Dumper=NoAliasDumper),
            f"{selected_config_path}",
            "text/yaml",
            key='download-yaml',
            width='stretch'
            )
        
        with col_save:
            if st.button("Save config", type="secondary", width='stretch'):
                st.write(selected_config_path)
                with open(selected_config_path,'w') as f:
                    yaml.dump(config_to_save, f, sort_keys=False, Dumper=NoAliasDumper)
    else:
        st.download_button(
            "Download Config",
            yaml.dump(config_to_save, sort_keys=False, Dumper=NoAliasDumper),
            f"{selected_config_path}",
            "text/yaml",
            key='download-yaml',
            width='stretch'
        )
        
def start_run_callback():
    st.session_state.is_running = True
    st.session_state.stop_requested = False

def stop_run_callback():
    st.session_state.stop_requested = True
    # st.toast("Experiment interruption requested. Waiting for current iteration to finish.")
 
def process_uploaded_results_csv():
    # Get the uploaded file from session state using the key.
    uploaded_file = st.session_state.csv_results_uploader
    
    if uploaded_file is not None:
        try:
            # To read the file as a string
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # string_data = stringio.read()
            csv_df = pd.read_csv(stringio)
            csv_dict = csv_df.to_dict(orient='records')
            # st.write('csv_dict', csv_dict)
            st.session_state.results = csv_dict
            # return csv_data
            # store the config in state for reset_config
            # st.session_state['csv_dict'] = csv_dict

            st.success(f"Successfully loaded configuration from '{uploaded_file.name}'!")
        
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")

def parse_round_info(df_columns):
    """
    Parses DataFrame columns to identify rounds, their types, factors, and response columns.
    
    Args:
        df_columns (list): A list of column names from the results DataFrame.

    Returns:
        dict: A dictionary where keys are round numbers and values are dicts
              containing 'type', 'factor_col', and 'response_col'.
              e.g., {0: {'type': 'scales', 'factor_col': '...', 'response_col': '...'}}
    """
    rounds_info = defaultdict(dict)
    # Regex to capture: 1: round number, 2: type, 3: rest of the name
    pattern = re.compile(r"round(\d+)_(\w+)_(\w+.*)")

    for col in df_columns:
        match = pattern.match(col)
        if match:
            round_num, round_type, name_part = match.groups()

            
            if 'raw' in round_type:
                continue

            round_num = int(round_num)
            
            rounds_info[round_num]['type'] = round_type

            if 'factor' in col:
                rounds_info[round_num]['factor_col'] = col
            elif 'response' in col:
                rounds_info[round_num]['response_col'] = col
                
    return dict(rounds_info)

def analyze_scales(df, model_col, factor_col, response_col):
    """Analyzes Likert scale data by calculating mean and std dev."""
    st.markdown(f"Grouping by `{model_col}` and `{factor_col}`. Aggregating `{response_col}`.")
    
    analysis_df = df.groupby([model_col, factor_col])[response_col].agg(['mean', 'std']).reset_index()
    analysis_df = analysis_df.round(2)

    st.dataframe(analysis_df)

    fig = px.bar(
        analysis_df,
        x=factor_col,
        y='mean',
        color=model_col,
        barmode='group',
        error_y='std',
        title=f"Mean Score by {factor_col.split('_')[-1].title()} and Model",
        labels={'mean': 'Mean Score (with Std Dev)', factor_col: factor_col.split('_')[-1].title()}
    )
    plotly_config = dict(
        width='stretch'
    )
    st.plotly_chart(fig, config=plotly_config)

def analyze_choice(df, model_col, factor_col, response_col):
    """Analyzes categorical choice data by counting occurrences."""
    st.markdown(f"Counting `{response_col}` grouped by `{model_col}` and `{factor_col}`.")

    analysis_df = df.groupby([model_col, factor_col, response_col]).size().reset_index(name='count')
    
    st.dataframe(analysis_df)

    fig = px.bar(
        analysis_df,
        x=factor_col,
        y='count',
        color=response_col,
        facet_col=model_col, # Creates separate charts for each model
        title=f"Choice Counts by {factor_col.split('_')[-1].title()} per Model",
        labels={'count': 'Count', factor_col: factor_col.split('_')[-1].title()}
    )
    plotly_config = dict(
        width='stretch'
    )
    st.plotly_chart(fig, config=plotly_config)

def analyze_ranking(df, model_col, factor_col, response_col):
    """Analyzes ranking data by calculating mean rank."""
    st.markdown(f"Analyzing mean rank of `{response_col}` grouped by `{model_col}` and `{factor_col}`.")
    
    # This analysis is identical to 'scales', so we can reuse the function
    analyze_scales(df, model_col, factor_col, response_col)

def run_analysis(df):
    st.write('# Analysis')
    
    rounds_info = parse_round_info(df.columns)
    if not rounds_info:
        st.error("Could not find any columns matching the 'round<N>_<type>_<name>' pattern.")
        return

    # Sort rounds by number to ensure chronological order
    for round_num in sorted(rounds_info.keys()):
        info = rounds_info[round_num]
        # st.write('info', info)
        round_type = info.get('type', 'unknown')
        round_type_clean = 'scales' if 'scales' in round_type.lower() else 'ranking' if 'ranking' in round_type.lower() else 'choice' if 'choice' in round_type.lower() else None
        factor_col = info.get('factor_col')
        response_col = info.get('response_col')

        with st.expander(f"### Round {round_num+1}: {round_type_clean.capitalize()}", expanded=True):
            if not factor_col or not response_col:
                st.warning(f"Skipping Round {round_num}. Missing factor or response column in data.")
                continue

            # Filter out rows where the necessary columns for this round are NaN
            # This is important because not all rows have data for all rounds
            round_df = df[['model_name', factor_col, response_col]].dropna()
            # st.dataframe(round_df)
            if round_df.empty:
                st.info(f"No data available for Round {round_num}.")
                continue

            # Dispatch to the correct analysis function based on type
            if round_type_clean == 'scales':
                analyze_scales(round_df, 'model_name', factor_col, response_col)
            elif round_type_clean == 'choice':
                analyze_choice(round_df, 'model_name', factor_col, response_col)
            elif round_type_clean == 'ranking':
                st.write('factor_col', factor_col)
                analyze_ranking(round_df, 'model_name', factor_col, response_col)
                
def show_results(df, selected_config_path):
    st.success(f"Experiment Run Complete.")

    st.write('---')
    st.write('# Results')
 
    st.dataframe(df, width='stretch')
    fn = Path(selected_config_path)
    fn = f'{fn.stem}.csv'
    st.download_button(
        "Download results",
        df.to_csv(index=False),
        fn,
        "text/csv",
        key='download-csv',
        width='stretch',
    )

def show_mixed_experiment_execution(selected_config_path):
    # The 'Start' button is only active when not running.
    if not st.session_state.is_running: 
        if st.button(
                "Start LLM Experiment Run", 
                type="primary", 
                width='stretch', 
                disabled=st.session_state.is_running,
                key='start_button_experiment'
            ):
            # check required API KEYS
            correct_keys = True
            missing_keys=[]
            if not st.session_state.test:
                selected_providers = set([m.split('-')[0] for m in st.session_state.models_to_test])
                for provider_name in selected_providers:
                    api_key = st.session_state['api_keys'][provider_name]
                    if (not api_key) or (api_key=='abc'):
                        correct_keys = False
                        missing_keys.append(provider_name)
        
            if correct_keys:
                st.session_state.is_running = True
                st.session_state.stop_requested = False
                st.rerun()
            else:
                st.error(f'You need to set your API keys for all providers to test.\n\n Missing keys for {missing_keys}')

    # The 'Stop' button is only shown while running.
    if st.session_state.is_running:
        if st.button(
                "STOP Experiment", 
                on_click=stop_run_callback,
                type="secondary", 
                width='stretch',
                key='stop_button_experiment'
            ):
            st.session_state.is_running = False
            st.session_state.stop_requested = True
            st.rerun()

    if st.session_state.is_running:
        st.session_state['results'] = []
        run_experiments()
        st.session_state.is_running = False
        st.rerun()


    with st.expander('Load Results'):
        st.file_uploader(
            "Upload a csv file for an experiment you already ran",
            type=['csv'],
            key="csv_results_uploader",  # A unique key is required for on_change
            on_change=process_uploaded_results_csv # The callback function
        )

    if st.session_state.get('results'):
        df = pd.DataFrame(st.session_state['results'])
        
        show_results(df, selected_config_path)

        run_analysis(df)
    
def remove_factor_from_state(factor_name, current_round):
    new_list = [f for f in current_round['factors_list'] if f[0]!=factor_name]
    st.toast(f'{factor_name} removed')
    current_round['factors_list'] = new_list

@st.dialog("Add Factor")
def show_add_factor_dialog(current_round, round_counter):
    df = pd.DataFrame({
        'Level Name': ["Name 1", "Name 2"],
        'Level Text for LLM': ["Text 1", "Text 2"]
    })
    new_factor_name = st.text_input('New Factor Name', key=f'new_factor_name_text_input_{current_round["round_type"]}_{round_counter}')
    edited_df = st.data_editor(
    df, 
    num_rows="dynamic",
    key=f'data_editor_{current_round["round_type"]}_{round_counter}'
    )
    new_factor_names_levels = edited_df['Level Name'].values.tolist()
    new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
    new_fact = [new_factor_name, [{f'{new_factor_name}|name':nf_name, f'{new_factor_name}|text':nf_text} for nf_name, nf_text in zip(new_factor_names_levels, new_factor_text_levels)]]
    if st.button("Add factor", key=f'add_button_{current_round["round_type"]}_{round_counter}'):
        current_round['factors_list'].append(new_fact)

        st.session_state.show_toast = True
        st.session_state.toast_text = f'Factor Added: {new_factor_name}'

        st.rerun()        



def show_factor_combinations_mixed(current_round, round_counter):
    with st.container(border=True):
        # --- Calculate all values first ---
        factor_levels = [fl[1] for fl in current_round['factors_list']]
        factor_names = [fl[0] for fl in current_round['factors_list']]
        factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        

        # st.write(factor_levels)
        # --- Display the information ---
        st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
        l = []
        for factor_name, factor_level in current_round['factors_list']:
            # st.write(factor_name)
            # st.write(factor_level)
            details = ', '.join([f[f'{factor_name}|name'] for f in factor_level])
            l.append(f'{factor_name} ({details})')
        factorial_text_display_details = ' x '.join(l)
        st.caption(f"{factorial_text_display_details}")
        

def show_factors_and_levels_mixed(current_round, round_counter):
    for factor_tuple in (current_round['factors_list']):
        factor_name = factor_tuple[0] # product
        factor_levels = factor_tuple[1] # [{product|name:'Name 1', product|text:'Text 1'},{product|name:'Name 2', product|text:'Text 2'}]
        # st.write('something')
        col_factor, col_remove = st.columns([2,1])
        with col_factor:
            st.write(factor_name)
        with col_remove:
            if st.button(f'Remove {factor_name}', width='stretch', key=f"remove_button_{current_round['round_type']}_{factor_name}"):
                remove_factor_from_state(factor_name, current_round=current_round)
                st.rerun()
        
        factor_levels_display = []
        for level_dict  in factor_levels:
            clean_dict = {k.split('|')[1]: v for k, v in level_dict.items()}
            factor_levels_display.append(clean_dict)

        original_df = pd.DataFrame(factor_levels_display)

        editor_key = f"data_editor_{current_round['round_type']}_{current_round['key']}_{factor_name}"

        st.data_editor(
            original_df, 
            num_rows='dynamic', 
            key=editor_key,
            on_change=apply_editor_changes_and_update_state,
            kwargs={
                    "factor_name": factor_name,
                    "state_list_to_update": current_round['factors_list'],
                    "editor_key": editor_key,
                    "original_df": original_df,  # Pass the original state to the callback
                    "has_group_id": False,  # Pass the original state to the callback
                },
            width='stretch',
            column_order=("name", "text"),
        )



def get_num_choices(current_round):
    # get number of choices
    num_choices = current_round['choices_shown_in_round']
    return num_choices


def mix_block_factor_variables(factors_products, block_variable_products):        
    full_products = [[*bvp, *fp] for fp in factors_products for bvp in block_variable_products]
    return full_products
    

def get_choice_factor_products(factors_list):        
    factor_levels = [fl[1] for fl in factors_list]
    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    return factor_products

def get_choice_permutations(factors_list, num_choices):
    if factors_list:
        factor_products = get_choice_factor_products(factors_list)
    else:
        factor_products = []
    combinations = list(itertools.permutations(factor_products, num_choices))
    combinations = [list(combo) for combo in combinations]
    return combinations

def show_choice_combinations_details_pro(current_round, round_counter):
    """
    Displays the experiment setup in a structured and professional card.
    """
    # st.subheader(f"Round {round_counter}: Experiment Setup")

    with st.container(border=True):
        # --- Calculate all values first ---
        factor_levels = [fl[1] for fl in current_round['factors_list']]
        factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        
        factor_products = get_choice_factor_products(current_round['factors_list'])
        total_combinations = len(factor_products)
        
        num_choices = get_num_choices(current_round)
        
        # Calculate the number of combinations (nCk)
        # Use math.comb for efficiency and clarity if available (Python 3.8+)
        # total_sets = math.comb(total_combinations, num_choices)
        combinations = get_choice_permutations(current_round['factors_list'], num_choices)
        total_sets = len(combinations)
        
        # --- Display the information ---
        st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Total Factor Combinations", 
                value=total_combinations,
                help="The total number of unique choices generated by the full factorial design. This is 'n'."
            )
        
        with col2:
            st.metric(
                label="Number of Choices Shown", 
                value=num_choices,
                help="The number of choices the LLM will be asked to evaluate in a single prompt. This is 'r'."
            )
            
        with col3:
            st.metric(
                label="Possible Choice Sets", 
                value=f"{total_sets:,}", # Adds a comma for thousands
                help=f"The total number of unique sets of {num_choices} items that can be created from the {total_combinations} total combinations."
            )

        st.caption(f"This is calculated as 'n choose r', or C({total_combinations}, {num_choices}).")

def show_ranking_combinations_details_pro(current_round, round_counter):

    with st.container(border=True):
        # --- Calculate all values first ---
        rank_permutations_multi_factor = get_rank_permutations_multi_factor(current_round['factors_list'])
        num_rank_permutations = len(rank_permutations_multi_factor)
        factor_levels = [fl[1] for fl in current_round['factors_list']]
        factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])
        
        
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Design", 
                value=f"{factorial_text_display} ",
                help="Full Factorial Design"
            )
            # st.markdown(f"**Design:** `{factorial_text_display}` Full Factorial Design")
        with col2:
            st.metric(
                label="Total Ranking Permutations", 
                value=num_rank_permutations,
                help="The total number of unique choices generated by the full factorial design. This is 'n'."
            )            

        # st.caption(f"This is calculated as 'n choose r', or C({total_combinations}, {num_choices}).")

    
    
def show_factor_items_ranking(current_round, round_counter):
    # st.write(current_round)
    if not current_round.get('factors_list'):
        num_items_in_ranking = 0
        st.info(f'No factors to rank. Add ***at least*** one factor.')
        if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
            show_add_factor_dialog(current_round, round_counter)
    else:
        num_items_in_ranking = len(current_round['factors_list'][0][1])
        st.write(f'{num_items_in_ranking} items to rank')
        
        show_ranking_combinations_details_pro(current_round, round_counter)
        
        show_factors_and_levels_mixed(current_round, round_counter)
        if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
            show_add_factor_dialog(current_round, round_counter)


def show_factor_items_choice(current_round, round_counter):
    
    if not current_round.get('factors_list'):
        st.info(f'No choice factors. Add at least one factor.')
        
        if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
            show_add_factor_dialog(current_round, round_counter)

        current_round['factors_list'] = []

    else:
        show_choice_combinations_details_pro(current_round, round_counter)


        st.write('---')
        show_factors_and_levels_mixed(current_round, round_counter)

        if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
            show_add_factor_dialog(current_round, round_counter)
        
        st.write('---')

        factor_products = get_choice_factor_products(current_round['factors_list'])
        max_possible_choices_shown = len(factor_products) if len(factor_products)>2 else 3
        choices_shown_in_round = st.slider('Number of Choices Shown (*r*)', min_value=2, max_value=max_possible_choices_shown, key=f'slider_{round_counter}_choice', help='''How many choices (r) to show to the LLM in this round. Maximum number of choices depends on n choose r (n: factor combination; r: choices shown to the LLM)''')

        if current_round['choices_shown_in_round'] != choices_shown_in_round:
            current_round['choices_shown_in_round'] = choices_shown_in_round
            st.rerun()
    

def show_factor_items_scales(current_round, round_counter):
    if not current_round.get('factors_list'):
        st.info(f'No scales factors. Add at least one factor.')
        current_round['factors_list'] = []

    else:
        show_factor_combinations_mixed(current_round, round_counter)
        st.write('---')
        show_factors_and_levels_mixed(current_round, round_counter)

    if st.button(f'Add {current_round["round_type"].capitalize()} Factor'):
        show_add_factor_dialog(current_round, round_counter)



def show_round_details(current_round, round_counter):
    '''
    '''
    if not current_round: return

    st.write(f'## Round {round_counter+1} - {current_round["round_type"].capitalize()}')
    # round_type = current_round['round_type']
    # round_metadata = Round_Types[round_type]

    tab_factor, tab_segment = st.tabs(
        ['Factors for this Round', 'Segment for this Round'],
    )
    with tab_factor:
        if current_round['round_type']=='ranking':
            show_factor_items_ranking(current_round, round_counter)
        elif current_round['round_type']=='choice':
            show_factor_items_choice(current_round, round_counter)
        elif current_round['round_type']=='scales':
            show_factor_items_scales(current_round, round_counter)

    with tab_segment:
        show_mixed_segments(current_round, round_counter)

    # st.write('---')
    if st.button('# Sample Text Shown to the LLM', width='stretch'):
        show_sample_mixed_modal(current_round, round_counter)

@st.dialog("Add New Round")
def show_add_round_modal():
    round_type = st.selectbox(
        'Select Round Type', 
        [rt.capitalize() for rt in Round_Types.keys()]
    ).lower()

    if st.button('Add Round', width='stretch'):
        
        new_round_key = str(uuid.uuid4())
        new_round = dict(
            key=new_round_key,
            segment='',
            factors_list=[],
            round_type=round_type
        )
        if round_type == 'choice':
            new_round['choices_shown_in_round'] = 2

        st.session_state.rounds.append(new_round)
        st.session_state.selected_round = new_round
        
        st.rerun()

def run_experiments():
    api_keys = st.session_state['api_keys']
    models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
    total_models_to_test = len(models_objects_to_test)


    rounds = st.session_state.get('rounds')
    if not rounds: return

    rounds_factor_combinations = get_rounds_factor_combinations()
    
    # st.write('len(rounds_factor_combinations)', len(rounds_factor_combinations))
    # st.write(rounds_factor_combinations)
    all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)
        
    # st.write('len(all_rounds_combinations)', len(all_rounds_combinations))
    # st.write('(all_rounds_combinations)', (all_rounds_combinations))
    
    block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations)
    
    # st.write('len(block_added_all_rounds_combinations_deduped)', len(block_added_all_rounds_combinations_deduped))

    
    total_iterations = total_models_to_test * len(block_added_all_rounds_combinations_deduped) * st.session_state.k

    progress_text = f"Running {total_iterations} experiments..."
    my_bar = st.progress(0, text=progress_text)
    progress_tracker = ProgressTracker(counter=0, progress_bar=my_bar, total_iterations=total_iterations)
    results = st.session_state['results']

    for model in models_objects_to_test:
        # go through all combinations
        go_through_all_combinations(
            block_added_all_rounds_combinations_deduped, 
            model=model, 
            results=results,
            progress_tracker=progress_tracker,
            )

    return results

def go_through_all_combinations(all_rounds_combinations, model, results, progress_tracker):
    for i, rounds_combination in enumerate(all_rounds_combinations):
        # st.write('# combination', i)
        # st.write('num rounds', len(rounds_combination))

        # do rounds_combination, k times
        k = st.session_state['k']
        run_combination_k_times(rounds_combination, k, model, results, progress_tracker)

def run_combination_k_times(rounds_combination, k, model, results, progress_tracker):
    for iteration in range(k):
        # st.write(f'## Iteration: {iteration}')
        run_combination(rounds_combination, iteration, model, results, progress_tracker)

def run_combination(rounds_combination, iteration, model, results, progress_tracker):
    # start round
    list_of_messages = []
    trial_data = {
            "trial_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "model_name": model.model_name,
            "iteration":iteration,
        }
    for j, (round_combination, current_round) in enumerate(zip(rounds_combination, st.session_state['rounds'])):
        
        time.sleep(st.session_state.sleep_amount)

        # st.write(f'- ### round {j}: {current_round["round_type"]}')
        # st.write(f'round_combination: ')
        # st.write(round_combination)

        if current_round["round_type"]=='ranking':
            # st.write('round_combination', round_combination)

            formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, round_combination) 
            if ranking_display_order:
                for ri, rank in enumerate(ranking_display_order):
                    for key, value in rank.items():
                        trial_data[f'round{j}_ranking_{ri}_{key}'] = value
                        # trial_data[f'round{j}_ranking_display_order'] = ranking_display_order

        elif current_round["round_type"]=='choice':
            # st.write('round_combination', round_combination)
            formatted_text, choices_display_order = get_llm_text_mixed_choice(round_combination, current_round)
            
            # st.write('formatted_text', formatted_text)
            # st.write('---')
            # st.write('---')

            if choices_display_order:
                for ci, choice in enumerate(choices_display_order):
                    for key, value in choice.items():
                        trial_data[f'round{j}_choice_{ci}_{key}'] = value

        elif current_round["round_type"]=='scales':
            formatted_text, factor_display = get_llm_text_mixed_scales(round_combination, current_round)
            # st.write(factor_display)
            # st.write('---')
            # st.write('---')
            if factor_display:    
                for key, value in factor_display.items():        
                    trial_data[f'round{j}_factor_{key}'] = value
        
        # st.write(formatted_text)
        list_of_messages.append({'role':'user','content':formatted_text}) # user_message
        
        llm_response, parsed_response = call_model_for_user_message(model, '', current_round, history=list_of_messages)
        # st.write(llm_response.content)
        # st.write(parsed_response)
        list_of_messages.append({'role':'assistant', 'content': llm_response.content})

        trial_data[f'round{j}_{current_round["round_type"]}_llm_response'] = parsed_response if parsed_response else None
        trial_data[f'round{j}_{current_round["round_type"]}_llm_raw_response'] = llm_response.content

        if current_round["round_type"]=='ranking':
            if parsed_response:
                try:
                    trial_data[f'round{j}_ranking_llm'] = [ranking_display_order[response-1] for response in parsed_response]
                except Exception as e:
                    trial_data[f'round{j}_ranking_llm'] = e
        elif current_round["round_type"]=='choice':
            if parsed_response:
                try:
                    trial_data[f'round{j}_choice_llm'] = choices_display_order[parsed_response-1]
                except Exception as e:
                    trial_data[f'round{j}_choice_llm'] = e
    results.append(trial_data)

    progress_tracker.counter+=1
    progress_made = int((progress_tracker.counter/progress_tracker.total_iterations)*100)
    progress_tracker.progress_bar.progress(progress_made, text=f"Experiments executed: {progress_tracker.counter}")
    # end round

@dataclass
class ProgressTracker:
    """Standardized response object for all LLM queries."""
    counter: int
    progress_bar: Any
    total_iterations: int


def call_model_for_user_message(model, user_message, current_round, history=[]):
    if st.session_state.test:
        if current_round['round_type']=='ranking':
            permutations = get_rank_permutations_multi_factor(current_round)
            num_to_rank = len(permutations[0])
            # st.write('num_to_rank', num_to_rank)
            llm_ranks = list(range(1, num_to_rank+1))
            random.shuffle(llm_ranks)
            llm_ranks_json = json.dumps(llm_ranks)

            response_data = LLMResponse(
                content=llm_ranks_json,
                response_time_ms=1,
                raw_response='raw_response'
            )
            parsed_choice = json.loads(llm_ranks_json)  

        elif current_round['round_type']=='choice':
            num_choices = current_round['choices_shown_in_round']
            choices_list = list(range(1, num_choices+1))
            llm_choice = random.choice(choices_list)
            llm_choice_json = json.dumps(llm_choice)

            response_data = LLMResponse(
                content=llm_choice_json,
                response_time_ms=1,
                raw_response='raw_response'
            )
            parsed_choice = json.loads(llm_choice_json)

        elif current_round['round_type']=='scales':
            llm_scales = random.choice(range(1, 8))
            llm_scales_json = json.dumps(llm_scales)
            
            response_data = LLMResponse(
                content=llm_scales_json,
                response_time_ms=1,
                raw_response='raw_response'
            )
            parsed_choice = json.loads(llm_scales_json)
    else: 
        response_data = model.query( 
            st.session_state.system_prompt, 
            user_message, 
            conversation_history=history
        )
    
        if 'error' in str(response_data.raw_response).lower():
            parsed_choice = None
        else:
            try:
                parsed_choice = json.loads(response_data.content)
            except Exception as e:
                parsed_choice = e
    return response_data, parsed_choice

def get_segment_text_from_segments_list(current_round):

    if not current_round['segment']:
        return []
    
    text_segment = current_round['segment']
    variables = re.findall(r"{(.*?)}", text_segment)
    variables = list(set(variables))
    # st.write('variables', variables)
    return variables

def filter_factors_list(factors_list, round_segment_variables):
    return [fac for fac in (factors_list) if fac[0] in round_segment_variables]


def group_id_in_segment_variables(round_segment_variables, block_variable_name, block_variable_level, block_variables):
    # st.write('block_variable_name::::::::::::', block_variable_name)
    # st.write('round_segment_variables', round_segment_variables)
    # st.write('block_variable_level::::::::::', block_variable_level)
    group_id = block_variable_level.get(f'{block_variable_name}|group_id')
    if not group_id:
        return []
    
    # st.write('group_id', group_id)
    # st.write('block_variables', block_variables)
    alias_levels = []
    for bvar_name, bvar_levels in block_variables:
        for bvar_level in bvar_levels:
            if ((bvar_level.get(f'{bvar_name}|group_id')==group_id) and (bvar_name in round_segment_variables)):
                alias_levels.append(bvar_level)
    # st.write(alias_levels)
    return alias_levels

def remove_dupes_from_complex_list(complex_list):
    # st.write('len(complex_list)', len(complex_list))
    new_list = []
    seen = set()

    for item in complex_list:
        key = json.dumps(item, sort_keys=True)
        if key not in seen:
            new_list.append(item)
            seen.add(key)
            
    # st.write('len(new_list)', len(new_list))
    
    return new_list

def get_rounds_factor_combinations():
    rounds = st.session_state.get('rounds')
    # st.write(rounds)

    rounds_factor_combinations = []
    for round_counter, current_round in enumerate(rounds):
        # st.write(round_counter)
        
        round_segment_variables = get_segment_text_from_segments_list(current_round)
        # st.write('round_segment_variables', round_segment_variables)

        filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
        # st.write(f'round {round_counter}','filtered_factors_list', filtered_factors_list)
        
        if current_round['round_type'] == 'ranking':
            factor_levels_rank_permutations = get_rank_permutations_multi_factor(filtered_factors_list)
            # st.write(factor_levels_rank_permutations)
            rounds_factor_combinations.append(factor_levels_rank_permutations)
        
        elif current_round['round_type'] == 'choice':
            num_choices = get_num_choices(current_round)
            combinations = get_choice_permutations(filtered_factors_list, num_choices)
            # st.write(combinations)
            rounds_factor_combinations.append(combinations)
            
        elif current_round['round_type'] == 'scales':
            factor_products = get_choice_factor_products(filtered_factors_list)
            rounds_factor_combinations.append(factor_products)
        
    return rounds_factor_combinations


def get_all_rounds_combinations(rounds_factor_combinations):
    all_rounds_combinations = list(itertools.product(*rounds_factor_combinations))
    all_rounds_combinations = [list(combo) for combo in all_rounds_combinations]
    return all_rounds_combinations

def get_block_added_all_rounds_combinations_deduped(all_rounds_combinations):

    rounds = st.session_state.get('rounds')
    
    block_variables = st.session_state.get('block_variables', [])
    
    if len(block_variables)<1:
        return all_rounds_combinations
    else:
        # use one factor (each factor is a different representation of the same construct)
        block_variable = block_variables[0]
        block_variable_name, block_variable_levels = block_variable
        
    # st.write('block_variables', block_variables)
    # st.write('all_rounds_combinations', all_rounds_combinations[:10])
    # return
    # st.write('len(all_rounds_combinations)', len(all_rounds_combinations))
    
    new_all_rounds_combinations = []
    for i, round_combination in enumerate(all_rounds_combinations):
        # st.write(f'combination {i}')
        for b, block_variable_level in enumerate(block_variable_levels): 
            # st.write(f'block {block_variable_level["country|name"]}')
            
            # insert each block_variable_level in the round_combination
            # same block_variable_level for all rounds
            new_round_combination = []
            for s, (single_round_combination, current_round) in enumerate(zip(round_combination, rounds)):
                # st.write(f'single_round_combination {s}')
                round_segment_variables = get_segment_text_from_segments_list(current_round)

                alias_levels = group_id_in_segment_variables(round_segment_variables, block_variable_name, block_variable_level, block_variables)
                # st.write(f"### {current_round['round_type']}")
                # st.write('block_variable_name', block_variable_name)
                # st.write('alias_levels', alias_levels)
                # if alias_levels:
                #     new_combination = [alias_levels, *combination]
                                
                if current_round['round_type']=='ranking': 
                    # list of lists (levels to rank)
                    # st.write('block_variable_name', block_variable_name)
                    # st.write('block_variable_level', block_variable_level)
                    # st.write('current_round', f"\n\n### **{current_round['round_type']}**")
                    # st.write(f'adding {block_variable_level["country|name"]}')
                    if alias_levels:
                        new_single_round_combination = [alias_levels, *single_round_combination]
                    else:
                        new_single_round_combination = single_round_combination[:]
                    # st.write('single_round_combination', single_round_combination)
                     
                    # return
                elif current_round['round_type']=='scales': 
                    # list of dict (factor levels)
                    if alias_levels:
                        new_single_round_combination = [*alias_levels, *single_round_combination]
                    else:
                        new_single_round_combination = single_round_combination[:]

                elif current_round['round_type']=='choice': 
                    # list of lists (choices_shown_in_round items)
                    if alias_levels:
                        new_single_round_combination = [alias_levels, *single_round_combination]
                    else:
                        new_single_round_combination = single_round_combination[:]

                    # st.write('block_variable_name', block_variable_name)
                    # st.write('block_variable_level', block_variable_level)
                    # st.write('current_round', f"\n\n### **{current_round['round_type']}**")
                    # st.write('single_round_combination', single_round_combination)
                    # return
                new_round_combination.append(new_single_round_combination)
            new_all_rounds_combinations.append(new_round_combination)
        # st.write('---')
    # st.write('len(new_all_rounds_combinations)', len(new_all_rounds_combinations))
    # st.write(new_all_rounds_combinations[:10])


    block_added_all_rounds_combinations_deduped = remove_dupes_from_complex_list(new_all_rounds_combinations)


    return block_added_all_rounds_combinations_deduped


def show_sample_ranking(current_round, round_counter):

    
    rounds = st.session_state.get('rounds')
    if not rounds: 
        st.warning('No Rounds. Add at least one round.')
        return

    rounds_factor_combinations = get_rounds_factor_combinations()
    
    all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)    
    # st.write(all_rounds_combinations)
    
    block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations)
    # st.write(block_added_all_rounds_combinations_deduped)
    # get a combination
    combo = block_added_all_rounds_combinations_deduped[0]
    # get current round from combination
    current_round_combo = combo[round_counter]
    
    
    if st.button('Refresh Sample', width='stretch' ,key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        combo = random.choice(block_added_all_rounds_combinations_deduped)
        # get current round from combination
        current_round_combo = combo[round_counter]
    
            
    formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, current_round_combo) 
    
    if not ranking_display_order:
        st.warning('No Ranking Factors')
    else:
        st.write('### Display Order')
        for i, rank_display in enumerate(ranking_display_order):
            st.write(f"{i+1}. {rank_display}" )
        st.write('---')
    
    st.write('### LLM Text')
    st.write(formatted_text) 
    st.write('---')
    

def show_sample_choice(current_round, round_counter):


    rounds = st.session_state.get('rounds')
    if not rounds: 
        st.warning('No Rounds. Add at least one round.')
        return

    rounds_factor_combinations = get_rounds_factor_combinations()
    
    # st.write('rounds_factor_combinations', rounds_factor_combinations)
    
    all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)
        
    # st.write('all_rounds_combinations', all_rounds_combinations)
    
    
    block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations)
    
    # st.write('block_added_all_rounds_combinations_deduped', block_added_all_rounds_combinations_deduped)
    
    
    # get a combination
    combo = block_added_all_rounds_combinations_deduped[0]
    # get current round from combination
    current_round_combo = combo[round_counter]
    
    # st.write([comb[round_counter] for comb in block_added_all_rounds_combinations_deduped])    

    if st.button('Refresh Sample', width='stretch' , key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        combo = random.choice(block_added_all_rounds_combinations_deduped)
        # get current round from combination
        current_round_combo = combo[round_counter]
    

    formatted_text, choices_display_order = get_llm_text_mixed_choice(current_round_combo, current_round) 

    if not choices_display_order:
        st.warning('No Choice Factors.')
    else:
        st.write('### Display Order')
        for i, choice_display in enumerate(choices_display_order):
            st.write(f"{i+1}. {choice_display}" )
        st.write('---')

    st.write('### LLM Text')
    st.write(formatted_text)
    st.write('---')

def show_sample_scales(current_round, round_counter):

    rounds = st.session_state.get('rounds')
    if not rounds: 
        st.warning('No Rounds. Add at least one round.')
        return

    rounds_factor_combinations = get_rounds_factor_combinations()
    # st.write('rounds_factor_combinations', rounds_factor_combinations)
    all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)    
    # st.write('all_rounds_combinations', all_rounds_combinations)
    
    block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations)
    # st.write('block_added_all_rounds_combinations_deduped', block_added_all_rounds_combinations_deduped)
    
    # get a combination
    combo = block_added_all_rounds_combinations_deduped[0]
    # get current round from combination
    current_round_combo = combo[round_counter]
    
    # st.write(current_round_combo )
    
    if st.button('Refresh Sample', width='stretch' , key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        combo = random.choice(block_added_all_rounds_combinations_deduped)
        current_round_combo = combo[round_counter]
        
    # st.write('current_round_combo', current_round_combo)
    formatted_text, factors_display = get_llm_text_mixed_scales(current_round_combo, current_round) 
    st.write(f'### Factor Combination: ')
    st.write(f'{factors_display}')
    st.write('---')
    st.write('### LLM Text')
    st.write(formatted_text)
    st.write('---')



def show_experiment_combinations():
    models_to_test = st.session_state.models_to_test
    total_models_to_test = len(models_to_test)

    rounds = st.session_state.get('rounds')
    if not rounds: return

    rounds_factor_combinations = get_rounds_factor_combinations()
    
    # st.write('len(rounds_factor_combinations)', len(rounds_factor_combinations))
    # st.write(rounds_factor_combinations)
    all_rounds_combinations = get_all_rounds_combinations(rounds_factor_combinations)
        
    # st.write('len(all_rounds_combinations)', len(all_rounds_combinations))
    # st.write('(all_rounds_combinations)', (all_rounds_combinations))
    
    block_added_all_rounds_combinations_deduped = get_block_added_all_rounds_combinations_deduped(all_rounds_combinations)
    
    # st.write('len(block_added_all_rounds_combinations_deduped)', len(block_added_all_rounds_combinations_deduped))
    # st.write('block_added_all_rounds_combinations_deduped', block_added_all_rounds_combinations_deduped)
    # all_combinations_length_display = ' x '.join([str(len(combo)) for combo in rounds_factor_combinations])
    
    total_iterations = total_models_to_test * len(block_added_all_rounds_combinations_deduped) * st.session_state.k
    k_value = st.session_state.k
    # Use a container to group the experiment summary
    with st.container(border=True):
        st.subheader("Experiment Configuration Summary")

        # Display the core parameters
        # st.markdown(f"**Combinations:** `{all_combinations_length_display}`")
        st.markdown(f"**K value:** `{k_value}`")

        # Use a divider to separate parameters from metrics
        st.divider()

        # Use columns for a dashboard-like layout of the key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Models", value=total_models_to_test)
        with col2:
            st.metric(label="Total Combinations", value=len(block_added_all_rounds_combinations_deduped))
        with col3:
            st.metric(label="Total Iterations", value=total_iterations)

def show_toast():
    if st.session_state.get('show_toast'):
        st.toast(st.session_state.get('toast_text'))
        st.session_state.show_toast = False

def show_round_container():
    st.write('## User Message Rounds')
    if not st.session_state.get('selected_round'):
        if not st.session_state.get('rounds'):
            st.session_state['selected_round'] = None
            st.session_state['selected_round_counter'] = None
        else:
            st.session_state['selected_round'] = st.session_state.rounds[0]
            st.session_state['selected_round_counter'] = 0

    with st.container(border=True):
        if not st.session_state.get('rounds'):
            st.info(f'**No Rounds:** Create at least one round of conversation for the LLM.')
            if st.button(":material/add: Add Round", width='stretch', type="secondary", key='add_round_modal_button_empty_round'):
                show_add_round_modal()

        else:
            col_master, col_details = st.columns(
                [1,3], 
                gap='medium',
                vertical_alignment='top',
                border=False
            )
            with col_master:
                st.write('## Rounds')
                row = st.columns([2, 1], gap='medium') # Create a small column for the delete button
                for round_counter, current_round in enumerate(st.session_state.rounds):
                    with row[0]: # round buttons
                        # Check if this is the selected round
                        is_selected = (current_round['key'] == st.session_state.selected_round['key'])
                        
                        # Set the button type accordingly
                        button_type = "primary" if is_selected else "secondary"

                        if st.button(
                            f'Round {round_counter+1} - {current_round["round_type"].capitalize()}',
                            key=f"round_btn_{current_round['key']}", # Always use a unique key for widgets in a loop
                            type=button_type,
                            width='stretch'
                        ):
                            st.session_state.selected_round = current_round
                            # Rerun to ensure the button style updates immediately
                            st.rerun() 
                        
                        if is_selected:
                            st.session_state['selected_round_counter'] = round_counter


                    with row[1]: # remove round
                        if st.button("✕", key=f"del_{current_round['key']}", help="Delete this round"):
                            removed_round = st.session_state.rounds.pop(round_counter)

                            if not st.session_state.rounds:
                                # Handle case where all rounds are deleted
                                st.session_state['selected_round'] = None
                                st.session_state['selected_round_counter'] = None
                                pass 
                            else:
                                st.session_state.selected_round = st.session_state.rounds[0]

                            st.session_state.show_toast = True
                            st.session_state.toast_text = f'Round {round_counter+1} - {current_round["round_type"].capitalize()} deleted'

                            st.rerun()

                with row[0]:                                
                    if st.button(":material/add: Add New Round", width='stretch', type="secondary", key='add_round_modal_button', help='Add New Round'):
                        show_add_round_modal()

            with col_details:
                with st.container(border=False):
                    show_round_details(st.session_state.get('selected_round'), st.session_state.get('selected_round_counter', 0))


def render_mixed_experiment(selected_config_path):
    show_toast()
    st.markdown("# Run a Behavioral Experiment on an LLM")
    st.markdown("**Select a configuration file, choose the LLMs, and modify the run parameters.**")
       
    st.file_uploader(
        "Upload a YAML configuration file for a predefined experiment.",
        type=['yaml', 'yml'],
        key="yaml_uploader",  # A unique key is required for on_change
        on_change=process_uploaded_yaml # The callback function
    )
    if st.button('Reset Configuration'):
        reset_config()

    with st.expander('System Prompt'):
        system_prompt = st.session_state.get('system_prompt', '')
        st.text_area(
            label='System Prompt', 
            value=system_prompt, 
            placeholder='''Type in your System Prompt''',
            key=f'system_prompt' 
        )

    show_experiment_configs_selectors()

    show_block_variables()
    # st.write(st.session_state.rounds)
    show_round_container()

    config_to_save = get_config_to_save()
    # st.write(config_to_save)

    show_experiment_combinations()

    show_download_save_config(config_to_save, selected_config_path)

    show_mixed_experiment_execution(selected_config_path)
   
     
def main():
    config_path = config_paths['mixed']
    config = load_experiment_config(config_path)

    # Store the current config path in session_state if needed later
    st.session_state.selected_config_path = config_path
        
    # On first render, Reset all relevant state variables from the new config
    for factor_to_load in config.keys():
        if st.session_state.get(factor_to_load) is None:
            st.session_state[factor_to_load] = config.get(factor_to_load)

    # Render the page
    render_mixed_experiment(selected_config_path=config_paths['mixed'])

def load_experiment_config(path_str):
    """Loads a YAML config file content from a given file path."""
    try:
        config_path = Path(path_str)
        with open(config_path, 'r', encoding='utf-8') as f:
            # Use yaml.safe_load for security
            config_dict = yaml.safe_load(f)
        return config_dict
    except FileNotFoundError:
        st.error(f"Configuration file not found at: {path_str}")
        return None
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML file: {e}")
        return None

if __name__ == "__main__":
    main()


