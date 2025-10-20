from .models import get_model, LLMResponse
import os
from pprint import pprint

from .global_configs import ALL_MODELS

provider_lookup = {
    'openai':'OPENAI_API_KEY',
    'anthropic':'ANTHROPIC_API_KEY',
    'google':'GEMINI_API_KEY',
    'deepinfra':'DEEPINFRA_API_KEY',
    'alibabacloud':'DASHSCOPE_API_KEY',    
}

for model_name in ALL_MODELS:
    parts = model_name.split('-', 1)
    assert len(parts) == 2, 'You should prefix model name with provider_name-'
    provider_key, api_model_name = parts
    api_key_env_var_name = provider_lookup[provider_key]    
    api_key = os.environ.get(api_key_env_var_name)

    api_keys = {
        provider_key: api_key
    }
    model = get_model(model_name, api_keys)
    print(model)
    # continue
    messages = [{"role": "user", "content": 'count to 5'}]

    response = model.query(
        '',
        '',
        conversation_history = messages
    )
    print('content', response.content)
    print(model_name, 'SUCCESSFUL')
    print('*'*20)