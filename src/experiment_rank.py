# src/experiment.py
import yaml
import pandas as pd
import itertools
import re
import json
import time
from pathlib import Path
import uuid
import random 
from tqdm import tqdm
from datetime import datetime
from models import get_model, LLMResponse

class ExperimentRunner:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        # print('self.config_path', self.config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.experiment_name = self.config_path.stem  # e.g., 'experiment_a'
        self.setup = self.config['choice_experiment_setup']
        self.test = self.config['test']
        self.rand_settings = self.config.get('randomization', {})
        self.results = []

    def _generate_all_profiles(self):
        """
        Generates all unique pairs of choices (products) for the experiment.
        """
        core_attrs = self.setup['factors_to_rank']
        attr_names = list(core_attrs.keys())
        level_name_lists = [list(attr['levels'].keys()) for attr in core_attrs.values()]
        level_value_lists = [list(attr['levels'].values()) for attr in core_attrs.values()]
        
        # 1. Create all unique product profiles from the factorial design (e.g., 2x2=4 profiles)
        all_profiles_names = [dict(zip(attr_names, combo)) for combo in itertools.product(*level_name_lists)]
        all_profiles_text = [dict(zip(attr_names, combo)) for combo in itertools.product(*level_value_lists)]
        
        # 2. Create all unique pairs of these profiles (e.g., C(4,2)=6 pairs)
        # print('level_name_lists', level_name_lists)
        # print('all_profiles', all_profiles)
        return all_profiles_names, all_profiles_text # 

    def _build_trial_text(self, block_level, block_variable_name, all_profiles, all_profiles_names):
        """
        Constructs the text description for one choice using core and randomized attributes.
        """
        template_parts = self.setup['choice_composition_template'] # list of placeholders
        # ["{product}", "{COO}", "{XYZ}"]
        # block_variable_items = self.setup['block_variables'][block_variable_name]['levels']
        if self.rand_settings.get('presentation_order', False):
            random.shuffle(all_profiles)

        num_choices = len(all_profiles)
        items=[]
        display_order=[]
        random_attribute_orders=[]
        for i_choice in range(num_choices): # num_choices options
            # option i
            option_text = '''---
**Option {choice_number}**
{choice_details}'''
            profile = all_profiles[i_choice]
            profile_name = all_profiles_names[i_choice]
            item = []
            choice_random_attributes=[]
            for part in template_parts:
                placeholders = re.findall(r"\{(\w+)\}", part)
                placeholder = placeholders[0]
                # print('profile', profile)
                if placeholder in self.setup['factors_to_rank']: # if placeholder is factors_to_rank
                    item.append(profile[placeholder])
                    display_order.append(profile_name[placeholder])
                elif placeholder in self.setup['block_variables']: # if placeholder is block_variables
                    item.append(block_level["value"])
                elif (self.setup['random_attributes']) and (placeholder in self.setup['random_attributes']): # if placeholder is block_variables
                    # print(placeholder)
                    dict_of_random_attribute = self.setup['random_attributes'][placeholder]['levels']
                    # {level_name: level_text}
                    level_name, level_text = random.choice(list(dict_of_random_attribute.items()))
                    choice_random_attributes.append(f'{placeholder}_{level_name}')
                    item.append(level_text)
            option_text = option_text.format(
                choice_number=i_choice+1,
                choice_details='\n'.join(item)
            )
            items.append(option_text)
            random_attribute_orders.append(choice_random_attributes)

        return items, display_order, random_attribute_orders

    def _parse_response(self, content):
        try:
            parsed_response = json.loads(content)
            return parsed_response
        except:
            return None

    def run(self):
        """Executes the choice experiment."""
        models_to_test = [get_model(name) for name in self.config['models_to_test']]
        all_profiles_names, all_profiles = self._generate_all_profiles() # all unique choice pairs
        print('all_profiles', all_profiles)
        print('all_profiles_names', all_profiles_names)

        k = self.config['k']
        sleep_amount = self.config['sleep_amount']
        # : [({'COO': 'Britain', 'XYZ': 'X'}, {'COO': 'Britain', 'XYZ': 'Y'}),...]
        block_variables = self.setup['block_variables'] #  {'product': {'values': [{'name': 'cars', 'value': 'car'}, ...}}

        print(f"--- Starting Factorial Choice Experiment: {self.experiment_name} ---")
        print(f"Factorial Design: {list(self.setup['factors_to_rank'].keys())}")
        # print(f"Total unique profiles: {len(list(itertools.product(*[attr['levels'] for attr in self.setup['factors_to_rank'].values()])))}")
        print(f"unique items to rank: {len(all_profiles)}")
        print(f"Randomization enabled: {self.rand_settings}")
        total_trials = len(models_to_test) * len(block_variables) * len(all_profiles) * k
        print('total_trials', total_trials)

        with tqdm(total=total_trials, desc="Running Choice Trials") as pbar:
            for model in models_to_test:
                for block_variable_name in block_variables.keys():
                    block_levels = block_variables[block_variable_name]['levels'] # [{'name': 'cars', 'value': 'car'}, {'name': 'food_products', 'value': 'food'},...]
                    for block_level in block_levels:
                        # START monte carlo sampling
                        for iteration in range(k):
                            trial_text_items, display_order, random_attribute_orders = self._build_trial_text(block_level, block_variable_name, all_profiles, all_profiles_names)
                            system_prompt = self.config['prompts']['system_prompt']
                            prompt_template = self.config['prompts']['user_prompt_template']
                            user_prompt = prompt_template.format(rank_choices='\n\n'.join(trial_text_items))
                            print('user_prompt', user_prompt)
                            if self.test:
                                response_data = LLMResponse(
                                    content='[2,1,3,4,5,6,7]',
                                    response_time_ms=1,
                                    raw_response='raw_response'
                                )
                            else:
                                response_data = model.query(system_prompt, user_prompt)
                            
                            parsed_rankings = self._parse_response(response_data.content)
                            # print('parsed_rankings', parsed_rankings)
                            rearranged_rankings = [display_order[i-1] for i in parsed_rankings]
                            # print('display', rearranged_rankings)
                            time.sleep(sleep_amount)

                            # 5. Log the results precisely as they were presented
                            trial_data = {
                                "trial_id": str(uuid.uuid4()),
                                "iteration":iteration,
                                "timestamp": datetime.now().isoformat(),
                                "model_name": model.model_name,
                                block_variable_name:block_level["value"], # product: car
                                'display_order':display_order,
                                'rearranged_rankings':rearranged_rankings,
                                "random_attribute_orders":random_attribute_orders,
                                "llm_rankings": parsed_rankings,
                                "raw_response": response_data.raw_response,
                            }

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