# src/experiment.py
import yaml
import pandas as pd
import itertools
from pathlib import Path
import re
import uuid
import time
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
        self.results = []

    def _generate_factorial_conditions(self):
        factor_names = list(self.setup['core_attributes'].keys())
        level_name_lists = [list(f['levels'].keys()) for f in self.setup['core_attributes'].values()]
        for combination in itertools.product(*level_name_lists):
            yield dict(zip(factor_names, combination))

    def _construct_scenario_prompt(self, condition):
        """Builds only the initial scenario part of the prompt."""
        prompt_parts = []
        composition_rules = self.config['prompts']['user_prompt_composition']

        for rule in composition_rules:
            if rule == "FACTORS_SECTION":
                for factor_name, level_name in condition.items():
                    stimulus_text = self.setup['core_attributes'][factor_name]['levels'][level_name]
                    prompt_parts.append(stimulus_text.strip())
            elif rule.startswith("component:"):
                component_name = rule.split(":", 1)[1]
                component_text = self.config['prompt_components'].get(component_name, "")
                prompt_parts.append(component_text.strip())
            else:
                prompt_parts.append(rule)
        # print('prompt_parts', prompt_parts)
        return "\n\n".join(prompt_parts)

    def _parse_response(self, raw_response):
        """Extracts the single digit from the LLM's response."""
        match = re.search(r'\d+', raw_response)
        if match:
            return int(match.group(0))
        return None

    def run(self):
        """Executes the full experiment defined in the config file."""
        models_to_test = [get_model(name) for name in self.config['models_to_test']]
        conditions = list(self._generate_factorial_conditions())
        k = self.config['k']
        sleep_amount = self.config['sleep_amount']

        # print(conditions)
        # return
        question_sequence = self.config['prompts']['question_sequence'] # list of questions to ask
        total_trials = len(models_to_test) * len(conditions) * k

        print(f"--- Starting Experiment: {self.experiment_name} ---")
        print(f"Factors: {list(self.setup['core_attributes'].keys())}")
        # print(f"Models: {[m.model_name for m in models_to_test]}")
        # print(f"conditions: {conditions}")
        print(f"Total conversational trials: {total_trials}")
        # return

        with tqdm(total=total_trials, desc="Running Trials") as pbar:
            for model in models_to_test:
                for condition in conditions:
                    # print(condition) # {'scarcity': 'high_scarcity', 'longevity': 'high_longevity'}
                    # run each condition k times
                    for iteration in range(k):
                        # START one trial
                        trial_data = {
                            "trial_id": str(uuid.uuid4()),
                            "iteration":iteration,
                            "timestamp": datetime.now().isoformat(),
                            "model_name": model.model_name,
                            **condition,
                        }
                        conversation_history = []
                        scenario_prompt = self._construct_scenario_prompt(condition)
                        # 2. Loop through questions
                        for i, question_details in enumerate(question_sequence):
                            user_prompt = f"{scenario_prompt}\n\n{question_details['question_text']}" if i == 0 else question_details['question_text']
                            system_prompt = self.config['prompts']['default_system_prompt']
                            
                            if self.test:
                                response_data = LLMResponse(
                                content='1',
                                response_time_ms=1,
                                raw_response='raw_response'
                            )
                                parsed_response = 1
                            else:
                                response_data = model.query(system_prompt, user_prompt, conversation_history)
                                parsed_response = self._parse_response(response_data.content)                            

                            time.sleep(sleep_amount)
                            
                            # Update conversation history for the next turn
                            conversation_history.append({"role": "user", "content": user_prompt})
                            conversation_history.append({"role": "assistant", "content": response_data.content})
                            
                            # print(f'round {i+1}',user_prompt)
                            # print('+'*20)
                            # 3. Add this question's data to the main trial dictionary with unique keys
                            response_key = question_details['response_key']
                            trial_data[f"{response_key}_parsed"] = parsed_response
                            trial_data[f"{response_key}_raw"] = response_data.raw_response
                        # 4. After all questions are asked, append the completed trial data to results
                        self.results.append(trial_data)
                        pbar.update(1)
                        self.save_results(final=False)
                        # END one trial



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