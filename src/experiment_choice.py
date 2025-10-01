# src/experiment.py
import yaml
import json
import pandas as pd
import itertools
import re
import time
from pathlib import Path
import uuid
import random 
import os
from tqdm import tqdm
from datetime import datetime
from models import get_model, LLMResponse
import streamlit as st
from utils import get_llm_text_choice, get_llm_text_rank, get_llm_text_scales
import openai



class StExperimentRunnerScales:
    def __init__(self, config_path=None, session_state=None):
        self.is_prod = "SPACE_ID" in os.environ
        self.config_path = Path(config_path)
        # self.config = config
        self.st_state = session_state if session_state else None
        self.results = []

    def call_model_for_user_message(self, model, user_message, history=[]):
        if st.session_state.test:
            response_data = LLMResponse(
                content='1',
                response_time_ms=1,
                raw_response='raw_response'
            )
            parsed_choice = 1
        else:
            response_data = model.query(
                st.session_state.system_prompt, 
                user_message, 
                conversation_history=history
            )
        
            if 'error' in str(response_data.raw_response).lower():
                parsed_choice = None
            else:
                parsed_choice = self._parse_response(response_data.content)
        return response_data, parsed_choice

    def run(self):
        """Executes the choice experiment."""
        api_keys = st.session_state['api_keys']
        models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
        total_models_to_test = len(models_objects_to_test)
        factor_combination_count = len(st.session_state.factor_products)
        total_iterations = st.session_state['k'] * factor_combination_count * total_models_to_test
        progress_text = f"Running {total_iterations} experiments..."
        my_bar = st.progress(0, text=progress_text)
        counter = 0
        for model in models_objects_to_test:
            # --- CRITICAL INTERRUPTION CHECK ---
            if (self.st_state) and (not self.st_state.is_running):
                st.warning(f"Experiment stopped by user.")
                break
            for i in range(st.session_state['k']):
                # --- CRITICAL INTERRUPTION CHECK ---
                if (self.st_state) and (not self.st_state.is_running):
                    st.warning(f"Experiment stopped by user.")
                    break
                for factor_product in st.session_state.factor_products:
                    # --- CRITICAL INTERRUPTION CHECK ---
                    if (self.st_state) and (not self.st_state.is_running):
                        st.warning(f"Experiment stopped by user.")
                        break
                    time.sleep(st.session_state['sleep_amount'])

                    # if st.session_state.randomize:
                    #     random.shuffle(choice_combination)
                    d = {k.split('|')[0]:v for factor in factor_product for k,v in factor.items()}
                    print(d)
                    print('*****************')
                    trial_data = {
                        "trial_id": str(uuid.uuid4()),
                        "iteration":i,
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model.model_name,
                    }
                    trial_data.update(d)
                    first_user_message, followup_text_list = get_llm_text_scales(factor_product)
                    # get response to the first query
                    response_data, parsed_choice = self.call_model_for_user_message(model, first_user_message, history=[])

                    trial_data["llm_response_1"] = parsed_choice
                    trial_data["raw_response_1"] = response_data.content
                    trial_data["user_message_1"] = first_user_message
                    
                    history = [
                        {'role':'user', 'content':first_user_message},
                        {'role':'assistant', 'content':response_data.content}
                    ]
                    # run the followup messages
                    if followup_text_list:
                        for jj, followup_text in enumerate(followup_text_list, 2):                            
                            response_data, parsed_choice = self.call_model_for_user_message(model, followup_text, history=history)
                            trial_data[f'llm_response_{jj}'] = parsed_choice
                            trial_data[f'raw_response_{jj}'] = response_data.content
                            trial_data[f'user_message_{jj}'] = followup_text
                            history.extend([
                                {'role':'user', 'content':followup_text},
                                {'role':'assistant', 'content':response_data.content}
                            ])                     
                        
                    self.results.append(trial_data)
                    self.save_results(final=False)
                    # st.write(st.session_state.system_prompt)
                    # st.write(user_message)
                    # st.write('******')
                    # st.write('******')
                    # st.write(progress_made)
                    progress_made = int((counter/total_iterations)*100)
                    my_bar.progress(progress_made, text=f"{counter} experiments executed...")
                    counter+=1
        my_bar.empty()
        self.save_results(final=True)

        # st.write(counter)
        return 
       
    def save_results(self, final=False):
        if not self.results:
            print("Warning: No results were generated.")
            return
        df = pd.DataFrame(self.results)
        
        config_dir = self.config_path.parent
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        output_path = config_dir/output_fn
        if not self.is_prod:
            df.to_csv(output_path, index=False)

    def get_results_df(self):
        if not self.results:
            print("Warning: No results were generated.")
            return None, None
        df = pd.DataFrame(self.results)
        
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        return df, output_fn

    def _parse_response(self, content):
        match = re.search(r'\b\d+\b', content)
        if match:
            return int(match.group(0))
        return None


class StExperimentRunnerRanking:
    def __init__(self, config_path=None, session_state=None):
        self.is_prod = "SPACE_ID" in os.environ
        self.config_path = Path(config_path)
        self.st_state = session_state if session_state else None
        self.results = []

    def run(self):
        """Executes the choice experiment."""
        api_keys = st.session_state['api_keys']
        models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
        total_models_to_test = len(models_objects_to_test)
        total_iterations = st.session_state['k'] * total_models_to_test
        if st.session_state.get('block_variable'):
            total_iterations*=len(st.session_state.get('block_variable')[1])
        progress_text = f"Running {total_iterations} experiments..."
        my_bar = st.progress(0, text=progress_text)
        counter = 0
        for model in models_objects_to_test:
            # --- CRITICAL INTERRUPTION CHECK ---
            if (self.st_state) and (not self.st_state.is_running):
                st.warning(f"Experiment stopped by user.")
                break
            for i in range(st.session_state['k']):
                # --- CRITICAL INTERRUPTION CHECK ---
                if (self.st_state) and (not self.st_state.is_running):
                    st.warning(f"Experiment stopped by user.")
                    break
                if st.session_state.get('block_variable'):
                    block_variable_name, block_variable_levels = st.session_state.get('block_variable')
                else:
                    block_variable_name, block_variable_levels = None, [None]
                for block_variable_level in block_variable_levels:
                    # --- CRITICAL INTERRUPTION CHECK ---
                    if (self.st_state) and (not self.st_state.is_running):
                        st.warning(f"Experiment stopped by user.")
                        break
                    factors_list = st.session_state['factors_list']
                    for iteration_factor, (factor_name, factor_levels) in enumerate(factors_list):
                        # --- CRITICAL INTERRUPTION CHECK ---
                        if (self.st_state) and (not self.st_state.is_running):
                            st.warning(f"Experiment stopped by user.")
                            break
                        factor_levels_copy = factor_levels[:]
                        if st.session_state.randomize:
                            random.shuffle(factor_levels_copy)
                        time.sleep(st.session_state['sleep_amount'])
                        user_message = get_llm_text_rank(factor_levels_copy, block_variable_level, block_variable_name)
                        if st.session_state.test:
                            mock_list = [1,2,3,4,5,6,7]
                            random.shuffle(mock_list)
                            response_data = LLMResponse(
                                content='1',
                                response_time_ms=1,
                                raw_response='raw_response'
                            )
                            parsed_choice = mock_list
                        else:
                            response_data = model.query(st.session_state.system_prompt, user_message)

                            if 'error' in str(response_data.raw_response).lower():
                                parsed_choice = None
                                return response_data.raw_response
                            else:
                                parsed_choice = self._parse_response(response_data.content)

                        factor_levels_display_order = [fl[f'{factor_name}|name'] for fl in factor_levels_copy] if parsed_choice else None
                        llm_ranking = [factor_levels_display_order[jj-1] for jj in parsed_choice] if parsed_choice else None
                        if block_variable_level:
                            block_variable_level_display = block_variable_level.get(f'{block_variable_name}|name') if parsed_choice else None
                        else:
                            block_variable_level_display = None
                        trial_data = {
                            "trial_id": str(uuid.uuid4()),
                            "iteration":i,
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model.model_name,
                            "iteration":i,
                            'factor_name':factor_name,
                            'block_variable_name':block_variable_name,
                            'block_variable_level_display':block_variable_level_display,
                            'factor_display_order':factor_levels_display_order,
                            "llm_ranking": llm_ranking,
                            "llm_ranking_int": parsed_choice,
                            "raw_response": response_data.content,
                        }
                            
                        self.results.append(trial_data)
                        self.save_results(final=False)
                        # st.write(st.session_state.system_prompt)
                        # st.write(user_message)
                        # st.write('******')
                        # st.write('******')
                        # st.write(progress_made)
                        progress_made = int((counter/total_iterations)*100)
                        my_bar.progress(progress_made, text=f"{counter} experiments executed...")
                        counter+=1
        my_bar.empty()
        self.save_results(final=True)

        return 
       
    def save_results(self, final=False):
        if not self.results:
            print("Warning: No results were generated.")
            return
        df = pd.DataFrame(self.results)
        
        config_dir = self.config_path.parent
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        output_path = config_dir/output_fn
        if not self.is_prod:
            df.to_csv(output_path, index=False)

    def get_results_df(self):
        if not self.results:
            print("Warning: No results were generated.")
            return None, None
        df = pd.DataFrame(self.results)
        
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        return df, output_fn

    def _parse_response(self, content):
        try:
            parsed = json.loads(content)
            return parsed
        except Exception  as e:
            print(e)
            return e


class StExperimentRunnerChoice:
    def __init__(self, config_path=None, session_state=None):
        self.is_prod = "SPACE_ID" in os.environ
        self.config_path = Path(config_path)
        # self.config = config
        self.st_state = session_state if session_state else None
        self.results = []

    def run(self):
        """Executes the choice experiment."""
        api_keys = st.session_state['api_keys']
        models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
        total_models_to_test = len(models_objects_to_test)
        choice_combination_count = len(st.session_state.combinations)
        total_iterations = st.session_state['k'] * choice_combination_count * total_models_to_test
        progress_text = f"Running {total_iterations} experiments..."
        my_bar = st.progress(0, text=progress_text)
        counter = 0
        for model in models_objects_to_test:
            # --- CRITICAL INTERRUPTION CHECK ---
            if (self.st_state) and (not self.st_state.is_running):
                st.warning(f"Experiment stopped by user.")
                break
            for i in range(st.session_state['k']):
                # --- CRITICAL INTERRUPTION CHECK ---
                if (self.st_state) and (not self.st_state.is_running):
                    st.warning(f"Experiment stopped by user.")
                    break
                for choice_combination in st.session_state.combinations:
                    # --- CRITICAL INTERRUPTION CHECK ---
                    if (self.st_state) and (not self.st_state.is_running):
                        st.warning(f"Experiment stopped by user.")
                        break
                    time.sleep(st.session_state['sleep_amount'])

                    if st.session_state.randomize:
                        random.shuffle(choice_combination)

                    user_message = get_llm_text_choice(choice_combination)
                    if st.session_state.test:
                        response_data = LLMResponse(
                            content='1',
                            response_time_ms=1,
                            raw_response='raw_response'
                        )
                        parsed_choice = 1
                    else:
                        response_data = model.query(st.session_state.system_prompt, user_message)
                        
                        if 'error' in str(response_data.raw_response).lower():
                            parsed_choice = None
                            return response_data.raw_response
                        else:
                            parsed_choice = self._parse_response(response_data.content)

                    trial_data = {
                        "trial_id": str(uuid.uuid4()),
                        "iteration":i,
                        "timestamp": datetime.now().isoformat(),
                        "model_name": model.model_name,
                        "llm_choice": parsed_choice,
                        "raw_response": response_data.content,
                    }
                    # st.write(choice_combination)
                    for i, choice_i in enumerate(choice_combination):
                        prefix = f"choice{i+1}"
                        for factor in choice_i:
                            key = list(factor.keys())[0].split('|')[0]
                            value = factor[f'{key}|name']
                            trial_data[f"{prefix}_{key}"] = value
                        
                    self.results.append(trial_data)
                    self.save_results(final=False)
                    # st.write(st.session_state.system_prompt)
                    # st.write(user_message)
                    # st.write('******')
                    # st.write('******')
                    # st.write(progress_made)
                    progress_made = int((counter/total_iterations)*100)
                    my_bar.progress(progress_made, text=f"{counter} experiments executed...")
                    counter+=1
        my_bar.empty()
        self.save_results(final=True)

        # st.write(counter)
        return 
       
    def save_results(self, final=False):
        if not self.results:
            print("Warning: No results were generated.")
            return
        df = pd.DataFrame(self.results)
        
        config_dir = self.config_path.parent
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        output_path = config_dir/output_fn
        if not self.is_prod:
            df.to_csv(output_path, index=False)

    def get_results_df(self):
        if not self.results:
            print("Warning: No results were generated.")
            return None, None
        df = pd.DataFrame(self.results)
        
        output_fn = f'{sanitize_filename(self.config_path.stem)}.csv'
        return df, output_fn

    def _parse_response(self, content):
        match = re.search(r'\b\d+\b', content)
        if match:
            return int(match.group(0))
        return None


class ExperimentRunner:
    def __init__(self, config_path=None, config=None, session_state=None):
        # print('config_path', config_path)
        # print('Path.cwd()', Path.cwd())
        self.config_path = Path(config_path)
        self.experiment_name = self.config_path.stem  # e.g., 'experiment_a'
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if config:
            self.config = config
            # self.experiment_name = config.get('experiment_name', 'Untitled Experiment')

        self.st_state = session_state if session_state else None
        self.is_interrupted = True

        self.setup = self.config['choice_experiment_setup']
        self.test = self.config.get('test', True)
        self.rand_settings = self.config.get('randomization', {})
        self.results = []

    def _generate_choice_sets(self):
        """
        Generates all unique pairs of choices (products) for the experiment.
        """
        core_attrs = self.setup['core_attributes']
        attr_names = list(core_attrs.keys())
        level_name_lists = [list(attr['levels'].keys()) for attr in core_attrs.values()]
        
        # 1. Create all unique product profiles from the factorial design (e.g., 2x2=4 profiles)
        all_profiles = [dict(zip(attr_names, combo)) for combo in itertools.product(*level_name_lists)]
        
        # 2. Create all unique pairs of these profiles (e.g., C(4,2)=6 pairs)
        choice_sets = list(itertools.combinations(all_profiles, 2))
        return choice_sets

    def _build_single_choice_description(self, core_profile, random_assignments):
        """
        Constructs the text description for one choice using core and randomized attributes.
        """
        template_parts = self.setup['choice_composition_template'] # list of placeholders
        description_parts = []
        
        # Create a single context dictionary for easy formatting
        # print('core_profile', core_profile)
        # print('random_assignments', random_assignments)
        context = {**core_profile, **random_assignments}
        # print(context)
        # print('template_parts', template_parts)
        for part in template_parts:
            # Find all placeholders like {brand}, {scarcity}, etc.
            placeholders = re.findall(r"\{(\w+)\}", part)
            
            formatted_part = part
            for placeholder in placeholders:
                # print('placeholder', placeholder)
 
                if placeholder in self.setup['core_attributes']:
                    # It's a core attribute, look up its text
                    level_name = context.get(placeholder)
                    if level_name:
                        value = self.setup['core_attributes'][placeholder]['levels'][level_name]
                    else:
                        value = "" # Should not happen in a valid setup
                elif placeholder in context:
                    # It's a randomized attribute, use its direct value
                    value = context[placeholder]
                    # print('value', value)
                else:
                    value = "" # Placeholder not found
                
                formatted_part = formatted_part.replace(f"{{{placeholder}}}", value.strip())
            
            description_parts.append(formatted_part)
            
        return "\n".join(description_parts)

    def _parse_response(self, content):
        print('*'*20)
        print('raw response:', content)
        print('*'*20)
        match = re.search(r'\b[1-2]\b', content)
        if match:
            return int(match.group(0))
        return None

    def run(self):
        """Executes the choice experiment."""
        models_to_test = [get_model(name) for name in self.config['models_to_test']]
        choice_sets = self._generate_choice_sets() # all unique choice pairs
        k = self.config['k']
        sleep_amount = self.config['sleep_amount']
        # for choice_set in choice_sets:
        #     print(choice_set)
        total_trials = len(models_to_test) * len(choice_sets) * k
        print('total_trials', total_trials)

        print(f"--- Starting Factorial Choice Experiment: {self.experiment_name} ---")
        print(f"Factorial Design: {list(self.setup['core_attributes'].keys())}")
        print(f"Total unique profiles: {len(list(itertools.product(*[attr['levels'] for attr in self.setup['core_attributes'].values()])))}")
        print(f"Total unique choice pairs: {len(choice_sets)}")
        print(f"Randomization enabled: {self.rand_settings}")

        with tqdm(total=total_trials, desc="Running Choice Trials") as pbar:
            for model in models_to_test:
                # --- CRITICAL INTERRUPTION CHECK ---
                if (self.st_state) and (not self.st_state.is_running):
                    st.warning(f"Experiment stopped by user.")
                    break
                for choice_set in choice_sets:
                    # --- CRITICAL INTERRUPTION CHECK ---
                    if (self.st_state) and (not self.st_state.is_running):
                        st.warning(f"Experiment stopped by user.")
                        break
                    profile_A, profile_B = choice_set

                    # 1. Randomly assign contextual attributes (e.g., brand names)
                    rand_assignments = {}
                    if 'randomized_attributes' in self.setup:
                        for rand_attr, config in self.setup['randomized_attributes'].items():
                            values = config['values'][:]
                            random.shuffle(values)
                            rand_assignments[rand_attr] = values                  

                    assignment_A = {k: v[0] for k, v in rand_assignments.items()}
                    assignment_B = {k: v[1] for k, v in rand_assignments.items()}

                    desc_A = self._build_single_choice_description(profile_A, assignment_A)
                    desc_B = self._build_single_choice_description(profile_B, assignment_B)
        
                    # 3. Randomize presentation order
                    presented_choices = [
                        {'profile': profile_A, 'assignments': assignment_A, 'description': desc_A},
                        {'profile': profile_B, 'assignments': assignment_B, 'description': desc_B}
                    ]
                    if self.rand_settings.get('presentation_order', False):
                        random.shuffle(presented_choices)

                    # 4. Construct the prompt
                    prompt_template = self.config['prompts']['user_prompt_template']
                    user_prompt = prompt_template.format(
                        choice_1_details=presented_choices[0]['description'],
                        choice_2_details=presented_choices[1]['description']
                    )

                    system_prompt = self.config['prompts']['system_prompt']
                    # print('user_prompt', user_prompt)
                    # print('*'*20)
                
                    # START monte carlo sampling
                    for iteration in range(k):
                        # --- CRITICAL INTERRUPTION CHECK ---
                        if (self.st_state) and (not self.st_state.is_running):
                            st.warning(f"Experiment stopped by user.")
                            break
                        # print('self.test', self.test)
                        if self.test:
                            response_data = LLMResponse(
                                content='1',
                                response_time_ms=1,
                                raw_response='raw_response'
                            )
                            parsed_choice = 1
                        else:
                            response_data = model.query(system_prompt, user_prompt)
                            parsed_choice = self._parse_response(response_data.content)
                        
                        time.sleep(sleep_amount)

                        # 5. Log the results precisely as they were presented
                        trial_data = {
                            "trial_id": str(uuid.uuid4()),
                            "iteration":iteration,
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model.model_name,
                            "llm_choice": parsed_choice,
                            "raw_response": response_data.raw_response,
                        }

                        for i, choice in enumerate(presented_choices):
                            prefix = f"choice{i+1}"
                            # Log core attributes
                            for attr, level in choice['profile'].items():
                                trial_data[f"{prefix}_{attr}"] = level
                            # Log randomized attributes
                            for attr, value in choice['assignments'].items():
                                trial_data[f"{prefix}_{attr}"] = value

                        self.results.append(trial_data)
                        self.save_results(final=False)
                        pbar.update(1)
                    # END monte carlo sampling


        self.save_results(final=True)

    def save_results(self, final=False):
        if not self.results:
            print("Warning: No results were generated.")
            return
        df = pd.DataFrame(self.results)
        
        config_dir = self.config_path.parent
        output_fn = f'{sanitize_filename(self.experiment_name)}.csv'
        output_path = config_dir/output_fn
        df.to_csv(output_path, index=False)
        if final:
            print(f"\n--- Experiment Finished ---")
            print(f"Results saved to {output_path}")

def sanitize_filename(name):
    # Function implementation
    return re.sub(r'[<>:"/\\|?*\s]', '_', name)