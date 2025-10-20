import os


APP_ENVIRONMENT = os.environ.get('APP_ENV', 'dev') 
is_prod = (APP_ENVIRONMENT == 'production')


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


default_config = {
    'system_prompt':'',
    'rounds':[],
    'block_variables':[],
    'k':1, 
    'test':True, 
    'sleep_amount':0.01,
    'models_to_test':[],
    'randomize':True,
    'paper_url':'',
    'api_keys':None
}
Round_Types = [
    'scales',
    'choice',
    'ranking',
]
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


config_paths = {
    'choice': 'config/choice/COO/country of origin choice.yaml',
    'scales': 'config/scales/ad_framing_and_scarcity.yaml',
    'ranking': 'config/rank/Balabanis 2004/Domestic Country Bias.yaml',
    'mixed': 'config/mixed/Domestic Country Bias.yaml',
}

