# src/experiment_runners.py
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
from utils import get_llm_text_mixed_rank, get_llm_text_mixed_choice, get_llm_text_mixed_scales


class StExperimentRunnerMixed:
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

    def get_round_results(self, query, history, round_counter):
        '''
        check for total rounds
        if round_counter >= total_rounds: finished
        else: 
            result = get_result(query, history)
            round_counter+=1
            get_round_results(query, history, round_counter)
        '''
        total_rounds = len(st.session_state.rounds)
        if round_counter>=total_rounds:
            return history
        result = query+'1'
        history.append({round_counter: result})
        round_counter+=1
        return self.get_round_results(result, history, round_counter)

    def run(self):
        """Executes the choice experiment."""
        api_keys = st.session_state['api_keys']
        models_objects_to_test = [get_model(name, api_keys) for name in st.session_state.models_to_test]
        total_models_to_test = len(models_objects_to_test)
        # total_rounds = len(st.session_state.rounds)
        total_iterations = st.session_state['k'] * total_models_to_test
        
        first_round = st.session_state.rounds[0]
        history = []
        final_result = self.get_round_results(first_round['key'], history, 0)
        print('final_result', final_result)

        self.results.append(final_result)
        self.save_results(final=False)
        #         
            # if current_round['round_type']=='ranking':
            #     factor_levels = current_round['factors_list'][0][1]
            #     factor_levels_permutations = list(itertools.permutations(factor_levels, len(factor_levels)))
            #     for factor_level_permutation in factor_levels_permutations:
            #         text_for_llm_sample, ranking_display_order = get_llm_text_mixed_rank(
            #             current_round, 
            #             factor_levels, 
            #             block_variable_level=None, 
            #             block_variable_name=None)
            #         print(factor_level_permutation)
            #         # print('text_for_llm_sample', text_for_llm_sample, 'ranking_display_order', ranking_display_order)
            # elif current_round['round_type']=='choice':
            #     print('nothing')

            
            # elif current_round['round_type']=='scales':
            #     print('nothing')

        # progress_text = f"Running {total_iterations} experiments..."
        # my_bar = st.progress(0, text=progress_text)
        # counter = 0
        # for model in models_objects_to_test:
        #     # --- CRITICAL INTERRUPTION CHECK ---
        #     if (self.st_state) and (not self.st_state.is_running):
        #         st.warning(f"Experiment stopped by user.")
        #         break
        #     for i in range(st.session_state['k']):
        #         # --- CRITICAL INTERRUPTION CHECK ---
        #         if (self.st_state) and (not self.st_state.is_running):
        #             st.warning(f"Experiment stopped by user.")
        #             break
        #         # query for each round
        #         trial_data = {
        #                     "trial_id": str(uuid.uuid4()),
        #                     "iteration":i,
        #                     "timestamp": datetime.now().isoformat(),
        #                     "model_name": model.model_name,
        #                 }
        #         for current_round in st.session_state.rounds:
        #             # --- CRITICAL INTERRUPTION CHECK ---
        #             if (self.st_state) and (not self.st_state.is_running):
        #                 st.warning(f"Experiment stopped by user.")
        #                 break
        #             time.sleep(st.session_state['sleep_amount'])

        #             if current_round['round_type']=='ranking':
                        
        #                 continue
        #                 # print('nothing')
        #             elif current_round['round_type']=='choice':
        #                 continue
        #                 # print('nothing')

                    
        #             elif current_round['round_type']=='scales':
        #                 continue
        #                 # print('nothing')



                    # if st.session_state.randomize:
                    #     random.shuffle(choice_combination)
                    # d = {k.split('|')[0]:v for factor in factor_product for k,v in factor.items()}
                    # print(d)
                    # print('*****************')
                    # trial_data = {
                    #     "trial_id": str(uuid.uuid4()),
                    #     "iteration":i,
                    #     "timestamp": datetime.now().isoformat(),
                    #     "model_name": model.model_name,
                    # }
                    # trial_data.update(d)
                    # first_user_message, followup_text_list = get_llm_text_scales(factor_product)
                    # # get response to the first query
                    # response_data, parsed_choice = self.call_model_for_user_message(model, first_user_message, history=[])

                    # trial_data["llm_response_1"] = parsed_choice
                    # trial_data["raw_response_1"] = response_data.content
                    # trial_data["user_message_1"] = first_user_message
                    
                    # history = [
                    #     {'role':'user', 'content':first_user_message},
                    #     {'role':'assistant', 'content':response_data.content}
                    # ]
                    # # run the followup messages
                    # if followup_text_list:
                    #     for jj, followup_text in enumerate(followup_text_list, 2):                            
                    #         response_data, parsed_choice = self.call_model_for_user_message(model, followup_text, history=history)
                    #         trial_data[f'llm_response_{jj}'] = parsed_choice
                    #         trial_data[f'raw_response_{jj}'] = response_data.content
                    #         trial_data[f'user_message_{jj}'] = followup_text
                    #         history.extend([
                    #             {'role':'user', 'content':followup_text},
                    #             {'role':'assistant', 'content':response_data.content}
                    #         ])                     
                        
                        
                    # progress_made = int((counter/total_iterations)*100)
                    # my_bar.progress(progress_made, text=f"{counter} experiments executed...")
                    # counter+=1

                # after all rounds -> save and update
        #         self.results.append(trial_data)
        #         self.save_results(final=False)
        #         progress_made = int((counter/total_iterations)*100)
        #         my_bar.progress(progress_made, text=f"{counter} experiments executed...")
        #         counter+=1
        # my_bar.empty()
        # self.save_results(final=True)

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

def sanitize_filename(name):
    # Function implementation
    return re.sub(r'[<>:"/\\|?*\s]', '_', name)