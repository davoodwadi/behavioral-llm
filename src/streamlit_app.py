import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
import json
import random
import uuid
from io import StringIO

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import os 
import time
from datetime import datetime

import threading

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
        
factors_to_load_dict = dict()
factors_to_load_dict['mixed'] = [
    'system_prompt',
    'rounds',
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'randomize',
    'paper_url',
]

if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'stop_requested' not in st.session_state:
    st.session_state.stop_requested = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'thread' not in st.session_state:
    st.session_state.thread = None

# Define config paths for each page
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
            # Update session state with the keys from the YAML file
            for state_var_name in yaml_data.keys():
                if state_var_name in yaml_data:
                    st.session_state[state_var_name] = yaml_data.get(state_var_name)
            
            st.success(f"Successfully loaded configuration from '{uploaded_file.name}'!")
        
        except Exception as e:
            st.error(f"Error processing YAML file: {e}")
# Function to stop the run (called by the button)

def show_experiment_configs_selectors():
    with st.expander('Experiment Configs'):
        st.session_state.k = st.number_input(
            "Number of Iterations", value=st.session_state.k, placeholder="Type a number..."
        )
        st.session_state.test = st.toggle('Run Mock Experiments', value=st.session_state.test)
        st.session_state.sleep_amount = st.number_input('Amount to pause between LLM API calls (seconds)', value=st.session_state.sleep_amount, )
        models_to_test = st.session_state.models_to_test
        filtered_models_to_test = [m for m in models_to_test if m in ALL_MODELS]
        if not filtered_models_to_test:
            filtered_models_to_test = ALL_MODELS[0]
        st.session_state.models_to_test = st.multiselect('LLMs to Test', ALL_MODELS, default=filtered_models_to_test)
        # Based on st.session_state.models_to_test
        # get required api keys
        if not st.session_state.test:
            selected_providers = set([m.split('-')[0] for m in st.session_state.models_to_test])
            for provider_name in selected_providers:
                # st.write(provider_name.capitalize())
                api_key = st.text_input(f'{provider_name.capitalize()} API key', type="password", key=f'{provider_name}_api_key_input')
                st.session_state['api_keys'][provider_name] = api_key
                
                # st.write(st.session_state['api_keys'])
        # 
        st.session_state.paper_url = st.text_input("Paper URL (optional)", value=st.session_state.paper_url)

        st.session_state.randomize = st.checkbox('Randomize Items Displayed in Ranking/Choice Segments', st.session_state.randomize)


def reset_config():
    # config_path = config_paths[st.session_state.page]
    config = load_experiment_config(st.session_state.selected_config_path)
    
    # Reset all relevant state variables from the new config
    for factor_to_load in config.keys():
        st.session_state[factor_to_load] = config.get(factor_to_load)
    # st.write(config['k'])
    st.rerun()
    
def show_segments(Segment_Types):
    for i, (segment) in enumerate(st.session_state.segments):
        col1_segments, col2_segments = st.columns([3,1])
        with col2_segments:
            with st.container(border=True):
                label_index = Segment_Types.index(st.session_state.segments[i].get('segment_label'))
                st.session_state.segments[i]['segment_label'] = st.selectbox(
                    'Change Segment Type',Segment_Types, 
                    index=label_index, 
                    key=f"type_selectbox_{segment['segment_id']}"
                )
                if st.button('Remove Segment', key=f"segment_remove_button_{segment['segment_id']}"):
                    st.write(f'segment removed: {segment["segment_id"]}')
                    remove_segment(segment["segment_id"])
        with col1_segments:
            st.session_state.segments[i]['segment_text'] = st.text_area(
                st.session_state.segments[i]['segment_label'], 
                value=st.session_state.segments[i]['segment_text'], 
                key=f"text_area_{segment['segment_id']}"
            )

def show_add_new_segment(Segment_Types):
    with st.container(border=True):
        col1_segments_add, col2_segments_add = st.columns([2,1], vertical_alignment='bottom')
        
        with col1_segments_add:
            New_Segment_Type = st.selectbox('Segment Type',Segment_Types, index=1, key=f"type_new_segment")
            
        with col2_segments_add:
            if st.button('Add Segment', width='stretch'):
                st.session_state.segments.append({
                'segment_label':New_Segment_Type,
                'segment_text':'',
                'segment_id':str(uuid.uuid1())[:5]
                })
                st.write('segment added')
                st.rerun()


def remove_factor(factor_name):
    st.toast(f'{factor_name} removed')
    new_list = [f for f in st.session_state.factors_list if f[0]!=factor_name]
    st.session_state.factors_list = new_list


def show_num_factor_items_to_rank():
    num_items_in_ranking = len(st.session_state.factors_list[0][1])
    st.write(f'{num_items_in_ranking} items to rank')


def show_factors_and_levels():
    for i, f in enumerate(st.session_state.factors_list):
        # st.write(f)
        factor_name = f[0] # product
        factor_levels = f[1] # [{product|name:'Name 1', product|text:'Text 1'},{product|name:'Name 2', product|text:'Text 2'}]
        
        col_factor, col_remove = st.columns([2,1])
        with col_factor:
            st.write(factor_name)
        with col_remove:
            st.button('Remove', on_click=remove_factor, args=[factor_name], width='stretch',key=factor_name)
        
        factor_levels_display = []
        for factor in factor_levels:
            new_dict = {k.split('|')[1]:v for k, v in factor.items()}
            factor_levels_display.append(new_dict)
        df = pd.DataFrame(factor_levels_display)
        # st.dataframe(df)
        edited_df = st.data_editor(df)
        edited_dict = edited_df.to_dict(orient='records')
        # st.write(edited_dict)

        edited_formatted_list = [factor_name, [{f"{factor_name}|name":level['name'], f"{factor_name}|text":level['text']} for level in edited_dict]]
        # st.write(edited_formatted_list)
        index = [j for j, fac in enumerate(st.session_state.factors_list) if fac[0]==factor_name][0]
        st.session_state.factors_list[index] = edited_formatted_list

def show_add_block_variable():
    with st.expander('Add Block Variable'):
        df = pd.DataFrame({
            'Level Name': ["Name 1", "Name 2", "Name 3"],
            'Level Text for LLM': ["Text 1", "Text 2", "Text 3"]
        })
        new_factor_name = st.text_input('New Block Variable Name', key='add_block_variable_name_text_input')
        edited_df = st.data_editor(
        df, 
        num_rows="dynamic", # Allows users to add/delete rows
        width=600,
        key='new_block_variable_data_editor'
        )
        new_factor_names_levels = edited_df['Level Name'].values.tolist()
        new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
        new_fact = [new_factor_name, [{f'{new_factor_name}|name':nf_name, f'{new_factor_name}|text':nf_text} for nf_name, nf_text in zip(new_factor_names_levels, new_factor_text_levels)]]
        # st.write(new_fact)
        if st.button("Add Block Variable", key = 'new_block_variable_button'):
            st.session_state.block_variable = new_fact
            st.rerun()
            st.write('Block Variable Added')

def remove_block_variable(variable_name):
    st.session_state.block_variable = ''
    st.toast(f'Block Variable: {variable_name} Removed.' )
    # st.rerun()

def show_block_variable():
    # st.write(st.session_state.get('block_variable'))
    if st.session_state.get('block_variable'):
        
        factor_name = st.session_state.block_variable[0] # product
        factor_levels = st.session_state.block_variable[1] # [{product|name:'Name 1', product|text:'Text 1'},{product|name:'Name 2', product|text:'Text 2'}]
        
        col_factor, col_remove = st.columns([2,1])
        with col_factor:
            st.write(factor_name)
        with col_remove:
            st.button('Remove', on_click=remove_block_variable, args=[factor_name], width='stretch',key=factor_name)
        
        factor_levels_display = []
        for factor in factor_levels:
            new_dict = {k.split('|')[1]:v for k, v in factor.items()}
            factor_levels_display.append(new_dict)
        df = pd.DataFrame(factor_levels_display)
        # st.dataframe(df)
        edited_df = st.data_editor(df)
        edited_dict = edited_df.to_dict(orient='records')
        # st.write(edited_dict)

        edited_formatted_list = [factor_name, [{f"{factor_name}|name":level['name'], f"{factor_name}|text":level['text']} for level in edited_dict]]
        st.session_state.block_variable = edited_formatted_list
        # st.write(st.session_state.block_variable)

    else:
        show_add_block_variable()

def show_add_factor():
    with st.expander('Add factor'):
        df = pd.DataFrame({
            'Level Name': ["Name 1", "Name 2", "Name 3"],
            'Level Text for LLM': ["Text 1", "Text 2", "Text 3"]
        })
        new_factor_name = st.text_input('New Factor ID', key='new_factor_name_text_input')
        edited_df = st.data_editor(
        df, 
        num_rows="dynamic", # Allows users to add/delete rows
        width=600
        )
        new_factor_names_levels = edited_df['Level Name'].values.tolist()
        new_factor_text_levels = edited_df['Level Text for LLM'].values.tolist()
        new_fact = [new_factor_name, [{f'{new_factor_name}|name':nf_name, f'{new_factor_name}|text':nf_text} for nf_name, nf_text in zip(new_factor_names_levels, new_factor_text_levels)]]
        # st.write(new_fact)
        if st.button("Add factor"):
            # print(new_fact)
            st.session_state.factors_list.append(new_fact)
            st.rerun()
            st.write('factor added')

def remove_segment(segment_id):
    st.session_state.segments = [seg for seg in st.session_state.segments if seg['segment_id']!=segment_id]
    st.rerun()

def remove_segment_from_round(segment_id):
    for i, r in enumerate(st.session_state.rounds):
        for j, segment in enumerate(r['segments']):
            if segment['segment_id']==segment_id:
                new_segments = [seg for seg in r['segments'] if seg['segment_id']!=segment_id]
                st.session_state.rounds[i]['segments'] = new_segments
                
    st.rerun()

def show_mixed_segments(Segment_Types, current_round, round_counter):
    st.info('Use `{factor_name}` to add placeholders for factors\n\nExample: {product} or {price}')

    for i, (segment) in enumerate(current_round['segments']):
        with st.container(border=True):
            col1_segments, col2_segments = st.columns([3,1])
            with col2_segments:
                with st.container(border=False):
                    label_index = Segment_Types.index(current_round['segments'][i].get('segment_label'))
                    current_round['segments'][i]['segment_label'] = st.selectbox(
                        'Change Segment Type',
                        Segment_Types, 
                        index=label_index, 
                        key=f"type_selectbox_{segment['segment_id']}"
                    )
                    if st.button('Remove Segment', key=f"segment_remove_button_{segment['segment_id']}"):
                        st.write(f'segment removed: {segment["segment_id"]}')
                        remove_segment_from_round(segment["segment_id"])
            with col1_segments:
                if ('Choice' in segment['segment_label']) or ('Ranking' in segment['segment_label']) or ('Treatment' in segment['segment_label']):
                    factor_names_raw = [f[0] for f in current_round['factors_list']]
                    factor_names_display = ['`{' + f + '}`' for f in factor_names_raw]
                    factor_names_str = ' '.join(factor_names_display)
                    # st.write('choice segment', factor_names)
                    text_content  = st.text_area(
                        f"{current_round['segments'][i]['segment_label']}\n\nRequired factors: {factor_names_str}", 
                        value=current_round['segments'][i]['segment_text'], 
                        key=f"{segment['segment_id']}"
                    )
                    current_round['segments'][i]['segment_text'] = text_content
                    # 3. Check for missing factors
                    missing_factors = []
                    # Loop through the raw factor names ('Sales', 'Marketing', etc.)
                    for factor in factor_names_raw:
                        # Create the placeholder string to search for ('{Sales}', '{Marketing}', etc.)
                        placeholder = f"{{{factor}}}"
                        if placeholder not in text_content:
                            missing_factors.append(factor)
                    if text_content and missing_factors:
                        # Format the missing factors for a clear warning message
                        missing_factors_formatted = ", ".join([f"`{{{f}}}`" for f in missing_factors])
                        st.warning(f"**Missing or incorrectly formatted factors:** {missing_factors_formatted}")
                else:
                    current_round['segments'][i]['segment_text'] = st.text_area(
                        current_round['segments'][i]['segment_label'], 
                        value=current_round['segments'][i]['segment_text'], 
                        key=f"{segment['segment_id']}"
                    )

def show_add_new_segment_to_round(Segment_Types, r):
    with st.container(border=True):
        col1_segments_add, col2_segments_add = st.columns([2,1], vertical_alignment='bottom')
        
        with col1_segments_add:
            New_Segment_Type = st.selectbox('New Segment Type', Segment_Types, index=1, key=f"type_new_segment_{r['key']}")
            
        with col2_segments_add:
            if st.button('Add Segment', width='stretch', key=f"add_new_segment_button_{r['key']}"):
                r['segments'].append({
                'segment_label':New_Segment_Type,
                'segment_text':'',
                'segment_id':str(uuid.uuid1())[:5]
                })
                st.write('segment added')
                st.rerun()


def get_config_to_save():
    factors_to_load = factors_to_load_dict['mixed']
    # config = load_experiment_config(st.session_state.selected_config_path)
    config_to_save = dict()
    for factor_to_load in factors_to_load:
        config_to_save[factor_to_load] = st.session_state.get(factor_to_load)
    return config_to_save

def show_sample_rank(current_round, round_counter):
    # st.write(f'### Round {index+1}')

    if st.session_state.get('block_variable'):
        block_variable_name, block_variable_levels = st.session_state['block_variable']
        block_variable_level = random.choice(block_variable_levels)
        # st.write(block_variable_name)
        # st.write(block_variable_level)
    else:
        block_variable_level = None
        block_variable_name = None

    # assume only 1 factor allowed for ranking

    factor_levels_rank_permutations = get_rank_permutations(current_round)
    # st.write(factor_levels_rank_permutations)
    
    if not factor_levels_rank_permutations:
        return
    
    factor_levels_rank_permutation = random.choice(factor_levels_rank_permutations)
    # st.write(factor_levels_rank_permutation)
    if st.button('Refresh Sample', key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        if st.session_state.randomize:
            random.shuffle(factor_levels_rank_permutation)
            if st.session_state.get('block_variable'):
                block_variable_level = random.choice(block_variable_levels)
            
    formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, factor_levels_rank_permutation, block_variable_level, block_variable_name) 
    
    st.write('### Display Order')
    for i, rank_display in enumerate(ranking_display_order):
        st.write(f"{i+1}. {rank_display}" )
    st.write('---')
    
    st.write('### LLM Text')
    st.write(formatted_text) 
    st.write('---')

def show_sample_choice(current_round, round_counter):
    # st.write(st.session_state.combinations[:2])
    combinations = get_choice_combinations(current_round)
    # st.write(len(combinations))
    if not combinations:
        return

    sample = random.choice(combinations)

    if st.session_state.randomize:
        random.shuffle(sample)

    if st.button('Refresh Sample', key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        sample = random.choice(combinations)
        if st.session_state.randomize:
            random.shuffle(sample)

    formatted_text, choices_display_order = get_llm_text_mixed_choice(sample, current_round) 
    st.write('### Display Order')
    for i, choice_display in enumerate(choices_display_order):
        st.write(f"{i+1}. {choice_display}" )
    st.write('---')

    st.write('### LLM Text')
    st.write(formatted_text)
    st.write('---')

def show_sample_scales(current_round, round_counter):
    factors_list = current_round.get('factors_list')
    
    if not factors_list:
        return 
    
    factor_products = get_choice_factor_products(factors_list)
    factor_product = random.choice(factor_products)

    if st.button('Refresh Sample', key=f'refresh_sample_{current_round["round_type"]}_{round_counter}'):
        factor_product = random.choice(factor_products)
    formatted_text, factors_display = get_llm_text_mixed_scales(factor_product, current_round) 
    st.write(formatted_text)
    st.write('---')


def show_sample_mixed(current_round, round_counter):
    # st.write(current_round)
    if current_round['round_type']=='ranking':
        show_sample_rank(current_round, round_counter)
    elif current_round['round_type']=='choice':
        show_sample_choice(current_round, round_counter)
    elif current_round['round_type']=='scales':
        show_sample_scales(current_round, round_counter)

def show_download_save_config(config_to_save, selected_config_path):
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
    if not is_prod:
        with col_save:
            if st.button("Save config", type="secondary", width='stretch'):
                st.write(selected_config_path)
                with open(selected_config_path,'w') as f:
                    yaml.dump(config_to_save, f, sort_keys=False, Dumper=NoAliasDumper)
        
def show_experiment_execution(selected_config_path, experiment_type='mixed'):
    # ----------------- Start Button Logic -----------------
    if st.button("Start LLM Experiment Run", type="primary", use_container_width=True, disabled=st.session_state.is_running):
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
            st.toast("Experiment started.")
            st.rerun()
        else:
            st.error(f'You need to set your API keys for all providers to test.\n\n Missing keys for {missing_keys}')
    # ----------------- Stop Button Logic (Only shown while running) -----------------
    if st.session_state.is_running:
        st.button(
            "STOP Experiment", 
            type="secondary", 
            use_container_width=True,
            on_click=stop_run_callback # Call the function to set the flag
        )    
    # ----------------- Execution Logic -----------------
    if st.session_state.is_running:
        if experiment_type=='mixed': 
            runner = StExperimentRunnerMixed(
                config_path=selected_config_path,
                session_state=st.session_state 
            )
        elif experiment_type=='choice':
            runner = StExperimentRunnerChoice(
                config_path=selected_config_path,
                session_state=st.session_state 
            )
        elif experiment_type=='ranking':
            runner = StExperimentRunnerRanking(
                config_path=selected_config_path,
                session_state=st.session_state 
            )
        elif experiment_type=='scales':
            runner = StExperimentRunnerScales(
                config_path=selected_config_path,
                session_state=st.session_state 
            )
        st.session_state.runner = runner
        
        error = runner.run()
        if error:
            st.error(f"Error in calling the LLM:\n\n{error}")
        else:
            # After run() completes or is interrupted, set the flag back to False
            st.session_state.is_running = False
        st.rerun() # Rerun to remove the spinner and stop button

    # ----------------- Success Logic -----------------
    if st.session_state.get('runner'):
        st.success(f"Experiment Run Complete.")

        results_df, fn = st.session_state.runner.get_results_df()
        if fn:
            st.subheader("Preliminary Results")
            st.dataframe(results_df)
            st.download_button(
                "Download results",
                results_df.to_csv(index=False),
                fn,
                "text/csv",
                key='download-csv',
                width='stretch',
            )

def start_run_callback():
    st.session_state.is_running = True
    st.session_state.stop_requested = False

def stop_run_callback():
    st.session_state.stop_requested = True
    # st.toast("Experiment interruption requested. Waiting for current iteration to finish.")
 
def show_mixed_experiment_execution(selected_config_path):
    # The 'Start' button is only active when not running.
    if not st.session_state.is_running: 
        if st.button(
                "Start LLM Experiment Run", 
                type="primary", 
                use_container_width=True, 
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
                use_container_width=True,
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




    if st.session_state.get('results'):
        st.success(f"Experiment Run Complete.")

        st.write('---')
        st.write('# Results')
        df = pd.DataFrame(st.session_state['results'])
        df.to_csv('./results.csv')
        st.dataframe(df, use_container_width=True)
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

def show_factor_items_ranking(current_round, round_counter):
    # st.write(current_round)
    if not current_round.get('factors_list'):
        num_items_in_ranking = 0
        st.info(f'No factors to rank. Add one factor.')
        show_add_factor_mixed(current_round, round_counter)

    else:
        num_items_in_ranking = len(current_round['factors_list'][0][1])
        st.write(f'{num_items_in_ranking} items to rank')
        show_factors_and_levels_mixed(current_round, round_counter)


def remove_factor_from_state(factor_name, current_round):
    new_list = [f for f in current_round['factors_list'] if f[0]!=factor_name]
    st.toast(f'{factor_name} removed')
    current_round['factors_list'] = new_list


def show_add_factor_mixed(current_round, round_counter):
    with st.expander(f'Add {current_round["round_type"]} factor'):
        df = pd.DataFrame({
            'Level Name': ["Name 1", "Name 2", "Name 3"],
            'Level Text for LLM': ["Text 1", "Text 2", "Text 3"]
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
            st.rerun()
            st.toast(f'Factor Added: {new_factor_name}')

def show_factor_combinations():
    factor_levels = [fl[1] for fl in st.session_state.factors_list]
    factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])

    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    st.session_state['factor_products'] = factor_products
    st.write(f'{len(factor_products)} Combinations: {factorial_text_display}')

def show_factor_combinations_mixed(current_round, round_counter):
    factor_levels = [fl[1] for fl in current_round['factors_list']]
    factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])

    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    current_round[f'factor_products'] = factor_products
    st.write(f'{len(factor_products)} Combinations. {factorial_text_display} Full Factorial.')

def show_factors_and_levels_mixed(current_round, round_counter):
    for i, f in enumerate(current_round['factors_list']):
        factor_name = f[0] # product
        factor_levels = f[1] # [{product|name:'Name 1', product|text:'Text 1'},{product|name:'Name 2', product|text:'Text 2'}]
        
        col_factor, col_remove = st.columns([2,1])
        with col_factor:
            st.write(factor_name)
        with col_remove:
            if st.button(f'Remove {factor_name}', width='stretch', key=f"remove_button_{current_round['round_type']}_{factor_name}"):
                remove_factor_from_state(factor_name, current_round=current_round)
                st.rerun()
        
        factor_levels_display = []
        for factor in factor_levels:
            new_dict = {k.split('|')[1]:v for k, v in factor.items()}
            factor_levels_display.append(new_dict)
        df = pd.DataFrame(factor_levels_display)
        edited_df = st.data_editor(df, num_rows='dynamic', key=f"data_editor_{current_round['round_type']}_{factor_name}")
        edited_dict = edited_df.to_dict(orient='records')

        edited_formatted_list = [factor_name, [{f"{factor_name}|name":level['name'], f"{factor_name}|text":level['text']} for level in edited_dict]]
        # st.write(edited_formatted_list)
        index = [j for j, fac in enumerate(current_round['factors_list']) if fac[0]==factor_name][0]
        current_round['factors_list'][index] = edited_formatted_list

def show_factor_items_ranking(current_round, round_counter):
    # st.write(current_round)
    if not current_round.get('factors_list'):
        num_items_in_ranking = 0
        st.info(f'No factors to rank. Add one factor.')
        show_add_factor_mixed(current_round, round_counter)

    else:
        num_items_in_ranking = len(current_round['factors_list'][0][1])
        st.write(f'{num_items_in_ranking} items to rank')
        show_factors_and_levels_mixed(current_round, round_counter)

def get_num_choices(current_round):
    # get number of choices
    num_choices = current_round['choices_shown_in_round']
    return num_choices

def get_choice_factor_products(factors_list):
    factor_levels = [fl[1] for fl in factors_list]
    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    return factor_products

def get_choice_combinations(current_round):
    num_choices = get_num_choices(current_round)
    factors_list = current_round.get('factors_list')
    if factors_list:
        factor_products = get_choice_factor_products(factors_list)
    else:
        factor_products = []
    combinations = list(itertools.combinations(factor_products, num_choices))
    combinations = [list(combo) for combo in combinations]
    return combinations

def get_rank_permutations(current_round):
    factors_list = current_round.get('factors_list')
    if not factors_list:
        return []
    factor_name, factor_levels_rank = factors_list[0]
    factor_levels_rank_permutations = list(itertools.permutations(factor_levels_rank, len(factor_levels_rank)))
    factor_levels_rank_permutations_list = [list(permutation) for permutation in factor_levels_rank_permutations]
    return factor_levels_rank_permutations_list

def show_choice_combinations_details(current_round, round_counter):
    num_choices = get_num_choices(current_round)
    factor_products = get_choice_factor_products(current_round['factors_list'])
    st.write(f"Number of choices for the LLM in this round: {num_choices}")
    total_combinations = len(factor_products)
    st.write(f"Total combinations: {total_combinations}")
    combinations = list(itertools.combinations(factor_products, num_choices))
    combinations = [list(combo) for combo in combinations]
    st.session_state['combinations'] = combinations
    st.write(f"{total_combinations} choose {num_choices}: *total {len(combinations)}*")

def show_factor_items_choice(current_round, round_counter):
    if not current_round.get('factors_list'):
        st.info(f'No choice factors. Add at least one factor.')
        current_round['factors_list'] = []
        show_add_factor_mixed(current_round, round_counter)

    else:
        show_factor_combinations_mixed(current_round, round_counter)
        show_choice_combinations_details(current_round, round_counter)
        show_factors_and_levels_mixed(current_round, round_counter)
        show_add_factor_mixed(current_round, round_counter)

def show_factor_items_scales(current_round, round_counter):
    if not current_round.get('factors_list'):
        st.info(f'No scales factors. Add at least one factor.')
        current_round['factors_list'] = []
        show_add_factor_mixed(current_round, round_counter)

    else:
        show_factor_combinations_mixed(current_round, round_counter)
        show_factors_and_levels_mixed(current_round, round_counter)
        show_add_factor_mixed(current_round, round_counter)

def show_round(current_round, round_counter):
    '''
    '''
    with st.expander(f'Round {round_counter+1} - {current_round["round_type"].capitalize()}'):
        round_type = current_round['round_type']
        round_metadata = Round_Types[round_type]
        round_segment_types = round_metadata['Segment_Types']

        if round_type=='choice':
            choices_shown_in_round = st.slider('Number of Choices Shown (*r*)', 2, key=f'slider_{round_counter}_{round_type}', help='''How many choices (r) to show to the LLM in this round. Maximum number of choices depends on n choose r (n: factor combination; r: choices shown to the LLM)''')
            current_round['choices_shown_in_round'] = choices_shown_in_round
        

        with st.expander(f"# Factors for this Round"):
            if current_round['round_type']=='ranking':
                show_factor_items_ranking(current_round, round_counter)
            elif current_round['round_type']=='choice':
                show_factor_items_choice(current_round, round_counter)
            elif current_round['round_type']=='scales':
                show_factor_items_scales(current_round, round_counter)

        with st.expander(f"# Segments for this Round"):

            show_mixed_segments(round_segment_types, current_round, round_counter)
            show_add_new_segment_to_round(round_segment_types, current_round)


        with st.expander(f'# Sample Text for LLM'):
            show_sample_mixed(current_round, round_counter)

        if st.button(f'Remove Round {round_counter+1} - {current_round["round_type"].capitalize()}', key=f'remove_round_{round_counter}', width='stretch'):
            # get round key
            key = current_round['key']
            st.session_state.rounds = [r for r in st.session_state.rounds if r['key']!=key]
            st.rerun()

def show_round_details(current_round, round_counter):
    '''
    '''
    # with st.expander(f'Round {round_counter+1} - {current_round["round_type"].capitalize()}'):
    st.write(f'Round Details: Round {round_counter+1} - {current_round["round_type"].capitalize()}')
    round_type = current_round['round_type']
    round_metadata = Round_Types[round_type]
    round_segment_types = round_metadata['Segment_Types']

    if round_type=='choice':
        choices_shown_in_round = st.slider('Number of Choices Shown (*r*)', 2, key=f'slider_{round_counter}_{round_type}', help='''How many choices (r) to show to the LLM in this round. Maximum number of choices depends on n choose r (n: factor combination; r: choices shown to the LLM)''')
        current_round['choices_shown_in_round'] = choices_shown_in_round
    

    with st.expander(f"# Factors for this Round"):
        if current_round['round_type']=='ranking':
            show_factor_items_ranking(current_round, round_counter)
        elif current_round['round_type']=='choice':
            show_factor_items_choice(current_round, round_counter)
        elif current_round['round_type']=='scales':
            show_factor_items_scales(current_round, round_counter)

    with st.expander(f"# Segments for this Round"):

        show_mixed_segments(round_segment_types, current_round, round_counter)
        show_add_new_segment_to_round(round_segment_types, current_round)


    with st.expander(f'# Sample Text for LLM'):
        show_sample_mixed(current_round, round_counter)

    # if st.button(f'Remove Round {round_counter+1} - {current_round["round_type"].capitalize()}', key=f'remove_round_{round_counter}', width='stretch'):
    #     # get round key
    #     key = current_round['key']
    #     st.session_state.rounds = [r for r in st.session_state.rounds if r['key']!=key]
    #     st.rerun()

def show_add_round():
    with st.expander('Add New Round'):
        col1, col2 = st.columns([1,1], vertical_alignment='bottom')
        with col1:
            round_type = st.selectbox('Round Type', [rt.capitalize() for rt in Round_Types.keys()], index=0, key='new round type selector').lower()

        with col2:
            if st.button('Add Round', key=f'button_add_round_', width='stretch'):
                all_keys = [int(ritem_value) for r in st.session_state.rounds for ritem_key, ritem_value in r.items() if ritem_key=='key']
                new_round_key = max(all_keys) + 1
                new_round = dict(
                    key = new_round_key,
                    segments = [],
                    factors_list = [],
                    round_type = round_type
                )
                if round_type == 'choice':
                    new_round['choices_shown_in_round'] = 2

                st.session_state.rounds.append(new_round)
                st.session_state.selected_round = new_round
                st.rerun()

@st.dialog("Add New Round")
def show_add_round_modal():
    round_type = st.selectbox(
        'Select Round Type', 
        [rt.capitalize() for rt in Round_Types.keys()]
    ).lower()

    if st.button('Add Round', width='stretch'):
        # --- Your existing logic for creating the round ---
        all_keys = [int(r['key']) for r in st.session_state.rounds] if st.session_state.rounds else [0]
        new_round_key = max(all_keys) + 1
        new_round = dict(
            key=new_round_key,
            segments=[],
            factors_list=[],
            round_type=round_type
        )
        if round_type == 'choice':
            new_round['choices_shown_in_round'] = 2

        st.session_state.rounds.append(new_round)
        st.session_state.selected_round = new_round
        
        # Close the dialog and rerun
        st.session_state['show_add_round_dialog'] = False
        st.rerun()

def run_experiments():
    api_keys = st.session_state['api_keys']
    models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
    total_models_to_test = len(models_objects_to_test)

    rounds = st.session_state.rounds

    rounds_factor_combinations = []
    for current_round in rounds:
        if current_round['round_type'] == 'ranking':
            factor_levels_rank_permutations = get_rank_permutations(current_round)
            rounds_factor_combinations.append(factor_levels_rank_permutations)
        elif current_round['round_type'] == 'choice':
            combinations = get_choice_combinations(current_round)
            rounds_factor_combinations.append(combinations)
        elif current_round['round_type'] == 'scales':
            factor_products = get_choice_factor_products(current_round['factors_list'])
            rounds_factor_combinations.append(factor_products)
            
    
    all_rounds_combinations = list(itertools.product(*rounds_factor_combinations))
    total_iterations = total_models_to_test * len(all_rounds_combinations) * st.session_state.k
    
    progress_text = f"Running {total_iterations} experiments..."
    my_bar = st.progress(0, text=progress_text)
    progress_tracker = ProgressTracker(counter=0, progress_bar=my_bar, total_iterations=total_iterations)
    results = st.session_state['results']

    for model in models_objects_to_test:
        # go through all combinations
        go_through_all_combinations(
            all_rounds_combinations, 
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

        if current_round["round_type"]=='ranking':
            formatted_text, ranking_display_order = get_llm_text_mixed_rank(current_round, round_combination, None, None) 
            if ranking_display_order:
                trial_data[f'round{j}_ranking_display_order'] = ranking_display_order
            # st.write('ranking_display_order', ranking_display_order)

        elif current_round["round_type"]=='choice':
            formatted_text, choices_display_order = get_llm_text_mixed_choice(round_combination, current_round)
            if choices_display_order:
                for ci, choice in enumerate(choices_display_order):
                    for key, value in choice.items():
                        trial_data[f'round{j}_choice_{ci}_{key}'] = value

        elif current_round["round_type"]=='scales':
            formatted_text, factor_display = get_llm_text_mixed_scales(round_combination, current_round)

            if factor_display:    
                for key, value in factor_display.items():        
                    trial_data[f'round{j}_factor_{key}'] = value
        
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
                    trial_data[f'round{j}_llm_rank'] = [ranking_display_order[response-1] for response in parsed_response]
                except Exception as e:
                    trial_data[f'round{j}_llm_rank'] = e
        elif current_round["round_type"]=='choice':
            if parsed_response:
                try:
                    trial_data[f'round{j}_llm_choice'] = choices_display_order[parsed_response-1]
                except Exception as e:
                    trial_data[f'round{j}_llm_choice'] = e
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
            num_to_rank = len(current_round['factors_list'][0][1])
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

def show_experiment_combinations():
    models_to_test = st.session_state.models_to_test
    total_models_to_test = len(models_to_test)

    rounds = st.session_state.rounds

    rounds_factor_combinations = []
    for current_round in rounds:
        if current_round['round_type'] == 'ranking':
            factor_levels_rank_permutations = get_rank_permutations(current_round)
            rounds_factor_combinations.append(factor_levels_rank_permutations)
        elif current_round['round_type'] == 'choice':
            combinations = get_choice_combinations(current_round)
            rounds_factor_combinations.append(combinations)
        elif current_round['round_type'] == 'scales':
            factor_products = get_choice_factor_products(current_round['factors_list'])
            rounds_factor_combinations.append(factor_products)
            
    all_combinations_length_display = ' x '.join([str(len(combo)) for combo in rounds_factor_combinations])
    
    all_rounds_combinations = list(itertools.product(*rounds_factor_combinations))
    total_iterations = total_models_to_test * len(all_rounds_combinations) * st.session_state.k
    k_value = st.session_state.k
    # Use a container to group the experiment summary
    with st.container(border=True):
        st.subheader("Experiment Configuration Summary")

        # Display the core parameters
        st.markdown(f"**Combinations:** `{all_combinations_length_display}`")
        st.markdown(f"**K value:** `{k_value}`")

        # Use a divider to separate parameters from metrics
        st.divider()

        # Use columns for a dashboard-like layout of the key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Total Models", value=total_models_to_test)
        with col2:
            st.metric(label="Total Combinations", value=len(all_rounds_combinations))
        with col3:
            st.metric(label="Total Iterations", value=total_iterations)

def show_toast():
    if st.session_state.get('show_toast'):
        st.toast(st.session_state.get('toast_text'))
        st.session_state.show_toast = False

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
        st.session_state.system_prompt = st.text_area(
            label='System Prompt', 
            value=st.session_state.system_prompt, 
            placeholder='''Type in your System Prompt''',
            key=f'mixed_system_prompt' 
        )

    show_experiment_configs_selectors()

    st.write('## User Message Rounds')
    if not st.session_state.get('selected_round'):
        st.session_state['selected_round'] = st.session_state.rounds[0]
    # st.write('selected round:', st.session_state['selected_round'])

    with st.container(border=True):
        col_master, col_details = st.columns(
            [1,2], 
            gap='large',
            vertical_alignment='top',
        )
        with col_master:
            row = st.columns([4, 1]) # Create a small column for the delete button

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
                        use_container_width=True # 'width' is deprecated, use 'use_container_width'
                    ):
                        st.session_state.selected_round = current_round
                        # Rerun to ensure the button style updates immediately
                        st.rerun() 
                    
                    if is_selected:
                        selected_round_counter = round_counter


                with row[1]: # remove round
                    if st.button("✕", key=f"del_{current_round['key']}", help="Delete this round"):
                        removed_round = st.session_state.rounds.pop(round_counter)

                        if not st.session_state.rounds:
                            # Handle case where all rounds are deleted
                            pass 
                        else:
                            st.session_state.selected_round = st.session_state.rounds[0]

                        st.session_state.show_toast = True
                        st.session_state.toast_text = f'Round {round_counter+1} - {current_round["round_type"].capitalize()} deleted'

                        st.rerun()
                        
            with row[0]:
                if st.button("Add New Round", use_container_width=True, type="secondary", key='add_round_modal_button'):
                    show_add_round_modal()

        with col_details:
            show_round_details(st.session_state['selected_round'], selected_round_counter)

    config_to_save = get_config_to_save()
    # st.write(config_to_save)
    show_download_save_config(config_to_save, selected_config_path)

    show_experiment_combinations()

    show_mixed_experiment_execution(selected_config_path)

   
     
def main():
    config_path = config_paths['mixed']
    config = load_experiment_config(config_path)

    # Store the current config path in session_state if needed later
    st.session_state.selected_config_path = config_path
        
    # On first render, Reset all relevant state variables from the new config
    for factor_to_load in config.keys():
        if not st.session_state.get(factor_to_load):
            st.session_state[factor_to_load] = config.get(factor_to_load)
    
    # st.write('is_prod', is_prod)

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


