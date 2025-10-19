import streamlit as st
import random 
import re
import pandas as pd
from pathlib import Path
from io import StringIO


def get_llm_text_mixed_rank(current_round, factor_levels):
    # st.write('factor_levels', factor_levels)
    
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
    filtered_factors_names = set([ff[0] for ff in filtered_factors_list])

    block_variables = st.session_state.get('block_variables')
    block_variables_names = set([block_variable[0] for block_variable in block_variables if block_variable])
    # st.write('block_variables_names', block_variables_names)
    block_factor_levels = []
    factor_factor_levels = []
    if block_variables:
        for factor_level in factor_levels:
            # get factor_level name
            key = set([fl['factor_name'] for fl in factor_level])
            # st.write('key', key)
            if key.intersection(block_variables_names):
                block_factor_levels.append(factor_level)
            elif key.issubset(filtered_factors_names):
                # st.write(f'{key} is subset of {filtered_factors_names}')
                factor_factor_levels.append(factor_level)
            else:
                st.warning(f"{key} is neither block variable nor is it used as a factor in the segment text")
                
    # st.write('block_factor_levels', block_factor_levels)
    # separate block and factors
    block_factor_dict = dict()
    block_factor_dict_display = dict()
    if block_factor_levels:
        for block_factor_level in block_factor_levels:
            # st.write('block_factor_level', block_factor_level)
            for item in block_factor_level:
                var_name = item['factor_name']
                block_factor_dict[var_name] = item['text']
                block_factor_dict_display[var_name] = item['name']
                
    text_for_llm_sample_list = []
    ranking_display_order = []
    
    for seg in current_round['segments']:
        if is_factor_in_segment(filtered_factors_names, seg):
            # repeat this segment
            for r, ranking_item in enumerate(factor_factor_levels, 1):
                fac = {item['factor_name']: item['text'] for item in ranking_item }
                fac_display = {item['factor_name']: item['name'] for item in ranking_item}
                fac.update(block_factor_dict)
                fac_display.update(block_factor_dict_display)
                
                ranking_display_order.append(fac_display)
                final_text = seg['segment_text'].format_map(SafeFormatter(fac))
                text_for_llm_sample_list.append(f"*Item {r}*:\n\n{final_text}")          
        else:
            # st.write('block_factor_dict', block_factor_dict)
            # st.write(seg['segment_text'])
            # no factor -> add it once
            # add block variables
            final_text = seg['segment_text'].format_map(SafeFormatter(block_factor_dict))
            
            text_for_llm_sample_list.append(final_text)   
    
    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)

    return text_for_llm_sample, ranking_display_order

def get_llm_text_mixed_choice(choice_combination, current_round):
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
    filtered_factors_names = set([ff[0] for ff in filtered_factors_list])

    block_variables = st.session_state.get('block_variables')

    block_variables_names = set([block_variable[0] for block_variable in block_variables if block_variable])
    block_factor_levels = []
    factor_factor_levels = []
    if block_variables:
        for choice_combo in choice_combination:
            # get factor_level name
            # st.write('choice_combo', choice_combo)
            key = set([fl['factor_name'] for fl in choice_combo])
            # st.write('key', key)
            if key.intersection(block_variables_names):
                block_factor_levels.append(choice_combo)
            elif key.issubset(filtered_factors_names):
                # st.write(f'{key} is subset of {filtered_factors_names}')
                factor_factor_levels.append(choice_combo)
            else:
                st.warning(f"{key} is neither block variable nor is it used as a factor in the segment text")
    
    text_for_llm_sample_list = []
    choices_display_order = []
    
    # separate block and factors
    block_factor_dict = dict()
    block_factor_dict_display = dict()
    if block_factor_levels:
        for block_factor_level in block_factor_levels:
            # st.write('block_factor_level', block_factor_level)
            for item in block_factor_level:
                # st.write('item', item)
                var_name = item['factor_name']
                block_factor_dict[var_name] = item['text']
                block_factor_dict_display[var_name] = item['name']
                
    # st.write(block_factor_dict_display)
    for seg in current_round['segments']:
            if is_factor_in_segment(filtered_factors_names, seg):
                # repeat this segment
                for c, choice in enumerate(factor_factor_levels, 1):
                    fac = {item['factor_name']: item['text'] for item in choice }
                    fac_display = {item['factor_name']: item['name'] for item in choice}
                    fac.update(block_factor_dict)
                    fac_display.update(block_factor_dict_display)
                    
                    choices_display_order.append(fac_display)
                    final_text = seg['segment_text'].format_map(SafeFormatter(fac))
                    text_for_llm_sample_list.append(f"*Choice {c}*:\n\n{final_text}")          
            else:
                # no factor -> add it once
                # add block variables
                final_text = seg['segment_text'].format_map(SafeFormatter(block_factor_dict))
                
                text_for_llm_sample_list.append(final_text)          

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample, choices_display_order

def is_factor_in_segment(filtered_factors_names, seg):
    for factor_name in list(filtered_factors_names):
        if '{'+factor_name+'}' in seg['segment_text']:
            return True
    return False
    
    
def get_llm_text_mixed_scales(factor_product, current_round):
    
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    
    filtered_factors_list = []
    filtered_factors_names = []
    fp_level_display = dict()
    fp_level = dict()
    for fp in factor_product:
        # does not run for empty factor product: no factor or block in segment 
        factor_name = fp['factor_name']
        fp_level.update({factor_name: fp['text']})
        fp_level_display.update({factor_name: fp['name']})
        if factor_name in round_segment_variables:
            filtered_factors_list.append(fp)
            filtered_factors_names.append(factor_name)
     
    text_for_llm_sample_list = []
    for seg in current_round['segments']:
        final_text = seg['segment_text'].format_map(SafeFormatter(fp_level))
        text_for_llm_sample_list.append(final_text)

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample, fp_level_display

def get_llm_text_choice(choice_combination):
    choice_segments = [segment for segment in st.session_state.segments if 'Choice' in segment['segment_label']]
    for choice, choice_segment in zip(choice_combination, choice_segments):
            # st.write(choice)
            extracted = {k.split('|')[0]:c[k] for c in choice for k in c.keys() if '|text' in k}
            # st.write(extracted)
            # st.write(choice_segment['segment_text'])
            choice_segment['segment_text_inserted'] = choice_segment['segment_text'].format(**extracted)
    text_for_llm_sample_list = []
    counter = 1   
    for segment in st.session_state.segments:
        if 'Choice' in segment['segment_label']:
            text = segment.get('segment_text_inserted')
            final_text = f'Choice {counter}\n\n{text}\n\n'
            counter+=1
        else:
            final_text = segment.get('segment_text')
        text_for_llm_sample_list.append(final_text)
    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample

def get_llm_text_rank(factor, block_variable_level=None, block_variable_name=None):
    # st.write(block_variable_level)
    # st.write(block_variable_name)
    # num_rankings = len(factor)
    ranking_segment = [segment for segment in st.session_state.segments if 'Ranking' in segment['segment_label']][0]
    segment_text = ranking_segment['segment_text']
    text_for_llm_sample_list = []

    counter = 1  
    # st.write(factor)
    for segment in st.session_state.segments:
        if 'Fixed' in segment['segment_label']:
            text_for_llm_sample_list.append(segment['segment_text']) 
        else:
            for factor_level in factor:
                factor_name = [k.split('|')[0] for k, v in factor_level.items()][0]
                fac = {factor_name:factor_level[f'{factor_name}|text']}
                if block_variable_name:
                    fac[block_variable_name] = block_variable_level[f'{block_variable_name}|text']
                text = segment_text.format(**fac)
                final_text = f"Item {counter}:\n\n{text}"
                counter+=1
                text_for_llm_sample_list.append(final_text)

    # st.write('---')
    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample

def get_llm_text_scales(factor_product):
    d = {k.split('|')[0]:v for factor in factor_product for k,v in factor.items()}
    first_text_for_llm_sample_list = []
    follow_up_questions = []
    counter=1
    for segment in st.session_state.segments:
        if 'Treatment' in segment['segment_label']:
            final_text = segment['segment_text'].format(**d)
            first_text_for_llm_sample_list.append(final_text)
        elif 'Question' in segment['segment_label']:
            if counter==1:
                first_text_for_llm_sample_list.append(segment['segment_text'])
                counter+=1
            else:
                follow_up_questions.append(segment['segment_text'])
        else:
            first_text_for_llm_sample_list.append(segment['segment_text'])

    text_for_llm_sample = '\n\n---\n\n'.join(first_text_for_llm_sample_list)
    return text_for_llm_sample, follow_up_questions

def get_segment_text_from_segments_list(current_round):

    if not current_round['segments']:
        st.write('No segments')
        return []
    
    text_segment = [seg['segment_text'] for seg in current_round['segments']]
    text_segment = '\n\n'.join(text_segment)
        
    variables = re.findall(r"{(.*?)}", text_segment)
    variables = list(set(variables))
    # st.write('variables', variables)
    return variables

def filter_factors_list(factors_list, round_segment_variables):
    return [fac for fac in (factors_list) if fac[0] in round_segment_variables]

class SafeFormatter(dict):
    def __missing__(self, key):
        return '{' + key + '}'  # Return the placeholder unchanged
    


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')
    
@st.cache_data(show_spinner="Processing results...", ttl=3600)
def create_dataframe_from_results(results_list):
    """Cache the DataFrame creation from results list"""
    return pd.DataFrame(results_list)


    
    

def process_uploaded_results_csv():
    # Get the uploaded file from session state using the key.
    uploaded_file = st.session_state.csv_results_uploader
    if uploaded_file is not None and uploaded_file!= st.session_state.get('last_uploaded_file'):
        
        try:
            # To read the file as a string
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            csv_df = pd.read_csv(stringio)
            csv_dict = csv_df.to_dict(orient='records')
            
            # This is the primary action: update the results
            st.session_state.results = csv_dict
            
            st.session_state.last_uploaded_file = uploaded_file

            # st.success(f"Successfully loaded results from '{uploaded_file.name}'!")
        
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            # Optional: Clear the tracker on error so the user can try uploading the same file again
            st.session_state.last_uploaded_file = None