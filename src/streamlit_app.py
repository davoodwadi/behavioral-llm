import streamlit as st
import pandas as pd
from pathlib import Path
import yaml
import json
import random
import uuid
from io import StringIO
import os 
import time
from experiment_choice import StExperimentRunnerChoice, StExperimentRunnerRanking, StExperimentRunnerScales
import itertools
from utils import get_llm_text_choice, get_llm_text_rank, get_llm_text_scales

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

is_prod = "SPACE_ID" in os.environ
# Initialize session state flags
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
# --- Utility Functions (Simulating your main_choice.py logic) ---

# Get the path to the 'config' directory (assuming it's relative to the app file)
APP_FILE = Path(__file__)
APP_DIR = APP_FILE.parent
PROJECT_DIR = APP_DIR.parent
CONFIG_DIR = PROJECT_DIR / 'config'
EXPERIMENT_TYPES = ['choice', 'scales', 'ranking']
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
factors_to_load_dict = dict()
factors_to_load_dict['choice'] = [
    'system_prompt', 
    'factors_list', 
    'segments', 
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'randomize',
    'paper_url',
]
factors_to_load_dict['ranking'] = [
    'system_prompt', 
    'factors_list',
    'block_variable', 
    'segments', 
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'randomize',
    'paper_url',
]
factors_to_load_dict['scales'] = [
    'system_prompt', 
    'factors_list', 
    'segments', 
    'k', 
    'test', 
    'sleep_amount',
    'models_to_test',
    'paper_url'
]
# Define config paths for each page
config_paths = {
    'choice': 'config/choice/COO/country of origin choice.yaml',
    'scales': 'config/scales/ad_framing_and_scarcity.yaml',
    'ranking': 'config/rank/Balabanis 2004/Domestic Country Bias.yaml' # Add other configs here
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

            # Update session state with the keys from the YAML file
            for state_var_name in factors_to_load_dict[st.session_state.page]:
                if state_var_name in yaml_data:
                    st.session_state[state_var_name] = yaml_data.get(state_var_name)
            
            st.success(f"Successfully loaded configuration from '{uploaded_file.name}'!")
        
        except Exception as e:
            st.error(f"Error processing YAML file: {e}")
# Function to stop the run (called by the button)
def stop_run_callback():
    st.session_state.is_running = False
    st.toast("Experiment interruption requested. Waiting for current iteration to finish.")
 
def show_experiment_configs_selectors():
    with st.expander('Experiment Configs'):
        st.session_state.k = st.number_input(
            "Number of Iterations", value=st.session_state.k, placeholder="Type a number..."
        )
        st.session_state.test = st.toggle('Run Mock Experiments', value=st.session_state.test)
        st.session_state.sleep_amount = st.number_input('Amount to pause between LLM API calls (seconds)', value=st.session_state.sleep_amount, )
        st.session_state.models_to_test = st.multiselect('LLMs to Test', ALL_MODELS, default=st.session_state.models_to_test)
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
        if st.session_state.page == 'ranking':
            st.session_state.randomize = st.checkbox('Randomize Items Displayed in Ranking', st.session_state.randomize)
        elif st.session_state.page == 'choice':
            st.session_state.randomize = st.checkbox('Randomize Items Displayed in Choices', st.session_state.randomize)

def remove_factor(factor_name):
    st.toast(f'{factor_name} removed')
    new_list = [f for f in st.session_state.factors_list if f[0]!=factor_name]
    st.session_state.factors_list = new_list

def get_choice_factor_products():
    factor_levels = [fl[1] for fl in st.session_state.factors_list]
    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    return factor_products

def show_factor_combinations():
    factor_levels = [fl[1] for fl in st.session_state.factors_list]
    factorial_text_display = " x ".join([str(len(fl)) for fl in factor_levels])

    factor_products = [list(p) for p in itertools.product(*factor_levels)]
    st.session_state['factor_products'] = factor_products
    st.write(f'{len(factor_products)} Combinations: {factorial_text_display}')

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
def show_segments(Segment_Types):
    for i, (segment) in enumerate(st.session_state.segments):
        # st.write(st.session_state.segments[i])
        col1_segments, col2_segments = st.columns([3,1])
        with col2_segments:
            label_index = Segment_Types.index(st.session_state.segments[i].get('segment_label'))
            st.session_state.segments[i]['segment_label'] = st.selectbox('Segment Type',Segment_Types, index=label_index, key=f"type_{segment['segment_id']}")
            if st.button('Remove Segment', key=f"segment_remove_button_{segment['segment_id']}"):
                st.write(f'segment removed: {segment["segment_id"]}')
                remove_segment(segment["segment_id"])
        with col1_segments:
            st.session_state.segments[i]['segment_text'] = st.text_area(
                st.session_state.segments[i]['segment_label'], 
                value=st.session_state.segments[i]['segment_text'], 
                key=f"{segment['segment_id']}"
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
    # st.write('---')

def get_num_choices():
    # get number of choices
    num_choices = len([seg for seg in st.session_state['segments'] if 'Choice' in seg['segment_label']])
    return num_choices
def show_choice_combinations_details():
    num_choices = get_num_choices()
    factor_products = get_choice_factor_products()
    st.write(f"Number of choices for the participant: {num_choices}")
    total_combinations = len(factor_products)
    st.write(f"Total combinations: {total_combinations}")
    combinations = list(itertools.combinations(factor_products, num_choices))
    combinations = [list(combo) for combo in combinations]
    st.session_state['combinations'] = combinations
    st.write(f"{total_combinations} choose {num_choices}: *total {len(combinations)}*")

def show_sample_choice(sample_to_show):
    # st.write(st.session_state.combinations[:2])
    with st.expander('### Text for LLM Sample'):
        samples = random.sample(st.session_state['combinations'], sample_to_show)
        if st.session_state.randomize:
            for sample in samples:
                random.shuffle(sample)

        if st.button('Refresh Sample'):
            samples = random.sample(st.session_state['combinations'], sample_to_show)

        for i, sample in enumerate(samples):
            st.write(f'### Sample {i+1}') 
            formatted_text = get_llm_text_choice(sample) 
            st.write(formatted_text)
            st.write('---')

def show_sample_scales(sample_to_show):
    # st.write(st.session_state.combinations[:2])
    with st.expander('### Text for LLM Sample'):
        factor_products = st.session_state.get('factor_products')
        factor_product = random.choice(factor_products)

        if st.button('Refresh Sample'):
            factor_product = random.choice(factor_products)

        first_formatted_text, followup_texts = get_llm_text_scales(factor_product) 
        st.write('### First Round')
        st.write(first_formatted_text)
        st.write('---')
        st.write('### Follow up Rounds:')
        for question in followup_texts:
            st.write(question)
        st.write('---')
def show_sample_rank():
    if st.session_state.get('block_variable'):
        block_variable_name, block_variable_levels = st.session_state.get('block_variable')
        block_variable_level = random.choice(block_variable_levels)
        # st.write(block_variable_name)
        # st.write(block_variable_level)
    else:
        block_variable_level = None
        block_variable_name = None
    with st.expander('### Text for LLM Sample'):
        factors_list = st.session_state['factors_list']
        for i, (factor_name, factor_levels) in enumerate(factors_list):
            factor_levels_copy = factor_levels[:]
            if i==0:
                if st.button('Refresh Sample'):
                    if st.session_state.randomize:
                        random.shuffle(factor_levels_copy)
                        if st.session_state.get('block_variable'):
                            block_variable_level = random.choice(block_variable_levels)
                
            st.write(f'### Sample {i+1}') 
            formatted_text = get_llm_text_rank(factor_levels_copy, block_variable_level, block_variable_name) 
            st.write(formatted_text)
            st.write('---')

def get_config_to_save():
    config_to_save = dict()
    for factor_to_load in factors_to_load_dict[st.session_state.page]:
        config_to_save[factor_to_load] = st.session_state[factor_to_load]
    return config_to_save

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
        
def show_experiment_execution(selected_config_path, experiment_type='choice'):
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
        if experiment_type=='choice':
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

def render_scales_experiment(selected_config_path):
    st.markdown("# Run a Scales Experiment")
    st.markdown("**Select a configuration file, choose the LLMs, and modify the run parameters.**")
       
    st.file_uploader(
        "Upload a YAML configuration file for a predefined experiment.",
        type=['yaml', 'yml'],
        key="yaml_uploader",  # A unique key is required for on_change
        on_change=process_uploaded_yaml # The callback function
    )

    with st.expander('System Prompt'):
        st.session_state.system_prompt = st.text_area(
            label='System Prompt', 
            value=st.session_state.system_prompt, 
            placeholder='''Type in your System Prompt''',
            key=f'scales_system_prompt' 
        )

    show_experiment_configs_selectors()

    with st.expander("# Factors"):
        show_factor_combinations()
        show_factors_and_levels()
        show_add_factor()

    with st.expander('### User Message'):
        st.info('Use `{factor_name}` to add placeholders for factors\n\nExample: {product} or {price}')

        Segment_Types = ['Segment', 'Treatment Segment', 'Question Segment']

        show_segments(Segment_Types)
        show_add_new_segment(Segment_Types)

    show_sample_scales(sample_to_show=1)

    config_to_save = get_config_to_save()
    # st.write(config_to_save)
    show_download_save_config(config_to_save, selected_config_path)
    show_experiment_execution(selected_config_path, experiment_type='scales')
   
def render_choice_experiment(selected_config_path):
    st.markdown("# Run a Choice Experiment")
    st.markdown("**Select a configuration file, choose the LLMs, and modify the run parameters.**")
       
    st.file_uploader(
        "Upload a YAML configuration file for a predefined experiment.",
        type=['yaml', 'yml'],
        key="yaml_uploader",  # A unique key is required for on_change
        on_change=process_uploaded_yaml # The callback function
    )

    with st.expander('System Prompt'):
        st.session_state.system_prompt = st.text_area(
            label='System Prompt', 
            value=st.session_state.system_prompt, 
            placeholder='''Type in your System Prompt''',
            key=f'choice_system_prompt' 
        )

    show_experiment_configs_selectors()

    with st.expander("# Factors"):
        show_factor_combinations()
        show_factors_and_levels()
        show_add_factor()

    with st.expander('### User Message'):
        st.info('Use `{factor_name}` to add placeholders for factors\n\nExample: {product} or {price}')

        Segment_Types = ['Fixed Segment', 'Choice Options']

        show_segments(Segment_Types)
        show_add_new_segment(Segment_Types)

    show_choice_combinations_details()
    st.write('---')

    show_sample_choice(sample_to_show=1)

    config_to_save = get_config_to_save()
    # st.write(config_to_save)
    show_download_save_config(config_to_save, selected_config_path)
    show_experiment_execution(selected_config_path, experiment_type='choice')
       
def render_ranking_experiment(selected_config_path):
    st.markdown("# Run a Ranking Experiment")
    st.markdown("**Select a configuration file, choose the LLMs, and modify the run parameters.**")    

    st.file_uploader(
        "Upload a YAML configuration file for a predefined experiment.",
        type=['yaml', 'yml'],
        key="yaml_uploader",  # A unique key is required for on_change
        on_change=process_uploaded_yaml # The callback function
    )

    with st.expander('System Prompt'):
        st.session_state.system_prompt = st.text_area(
            label='System Prompt', 
            value=st.session_state.system_prompt, 
            placeholder='''Type in your System Prompt''',
            key=f'ranking_system_prompt' 
        )
    
    show_experiment_configs_selectors()
    
    with st.expander("# Factor to Rank"):
        show_num_factor_items_to_rank()
        show_factors_and_levels()
        show_add_factor()

    with st.expander("# Block Variable"):
        show_block_variable()

    with st.expander('### User Message'):
        st.info('Use `{factor name}` and `{block variable name}` to add placeholders for their items.\n\nExample: You are shopping for a {product} from {price}')

        Segment_Types = ['Fixed Segment', 'Ranking Segment']

        show_segments(Segment_Types)
        show_add_new_segment(Segment_Types)

    show_sample_rank()

    config_to_save = get_config_to_save()
    # st.write(config_to_save)
    show_download_save_config(config_to_save, selected_config_path)
    show_experiment_execution(selected_config_path, experiment_type='ranking')
     
def main():
    # Set up session state for simple navigation
    DEFAULT_PAGE = 'choice'
    page_index = EXPERIMENT_TYPES.index(DEFAULT_PAGE)
    if 'page' not in st.session_state:
        st.session_state.page = EXPERIMENT_TYPES[page_index]
        # print('first render of the app')
        config_path = config_paths[st.session_state.page]
        config = load_experiment_config(config_path)
        
        # Reset all relevant state variables from the new config
        for factor_to_load in factors_to_load_dict[st.session_state.page]:
            st.session_state[factor_to_load] = config.get(factor_to_load)
        
        # Store the current config path in session_state if needed later
        st.session_state.selected_config_path = config_path

    # Sidebar Navigation
    with st.sidebar:
        previous_page = st.session_state.page

        navigation_choice = st.radio(
            "Select your experiment type:",
            [et.capitalize() for et in EXPERIMENT_TYPES],
            index=EXPERIMENT_TYPES.index(st.session_state.page),
        )
        st.session_state.page = navigation_choice.lower()

        st.markdown("---")
        st.markdown("Developed for the Behavioral LLM Manuscript.")
    
    # If the page has changed since the last run, reload the config.
    # This is the core logic that replaces the sidebar_changed flag.

    if previous_page != st.session_state.page:
        # print(f"Page changed from '{previous_page}' to '{st.session_state.page}'. Reloading config.")
    
        config_path = config_paths[st.session_state.page]
        config = load_experiment_config(config_path)
        
        # Reset all relevant state variables from the new config
        for factor_to_load in factors_to_load_dict[st.session_state.page]:
            # print(factor_to_load, 'changed to:', config.get(factor_to_load))
            st.session_state[factor_to_load] = config.get(factor_to_load)
        
        # Store the current config path in session_state if needed later
        st.session_state.selected_config_path = config_path

        # We can force a rerun here to ensure a clean slate, though it's often not necessary
        st.rerun()

    # Render the selected page
    if st.session_state.page == 'choice':
        render_choice_experiment(selected_config_path=config_paths['choice'])
    elif st.session_state.page == 'ranking':
        render_ranking_experiment(selected_config_path=config_paths['ranking'])
    elif st.session_state.page == 'scales':
        render_scales_experiment(selected_config_path=config_paths['scales'])

# Define a cached function to load the YAML content
# Caching is crucial here to prevent re-reading the file on every app rerun (widget interaction).
# @st.cache_data(show_spinner="Loading experiment configuration...")
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

@st.cache_data
def get_available_configs():
    """Scans the config directory for available experiment files."""
    available_configs = []
    if not CONFIG_DIR.is_dir():
        st.error(f"Error: Config directory not found at '{CONFIG_DIR}'")
        return available_configs

    for experiment_type in EXPERIMENT_TYPES:
        type_dir = CONFIG_DIR / experiment_type
        if type_dir.is_dir():
            for config_file_path in type_dir.rglob('*.yaml'):
                # Create user-friendly display name: type / folder / filename
                relative_path = config_file_path.relative_to(CONFIG_DIR)
                parts = [Path(p).stem if Path(p).suffix=='.yaml' else p for p in relative_path.parts]
                display_name = ' | '.join(parts)
                available_configs.append((experiment_type, display_name, config_file_path))

    # Sort files alphabetically
    available_configs.sort(key=lambda x: x[1])
    return available_configs

if __name__ == "__main__":
    main()

