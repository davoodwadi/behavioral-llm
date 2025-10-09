import streamlit as st
import random 
import re

def get_llm_text_mixed_rank(current_round, factor_levels):
    # st.write('factor_levels', factor_levels)
    
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
    filtered_factors_names = set([ff[0] for ff in filtered_factors_list])
    # st.write('round_segment_variables', round_segment_variables)
    # st.write('filtered_factors_names', filtered_factors_names)
    # st.write('filtered_factors_list', filtered_factors_list)

    block_variables = st.session_state.get('block_variables')

    block_variables_names = set([block_variable[0] for block_variable in block_variables if block_variable])
    # st.write('block_variables_names', block_variables_names)
    block_factor_levels = []
    factor_factor_levels = []
    if block_variables:
        for factor_level in factor_levels:
            # get factor_level name
            key = set([k.split('|')[0] for fl in factor_level for k,v in fl.items()])
            # st.write('key', key)
            if key.intersection(block_variables_names):
                block_factor_levels.append(factor_level)
            elif key.issubset(filtered_factors_names):
                # st.write(f'{key} is subset of {filtered_factors_names}')
                factor_factor_levels.append(factor_level)
            else:
                st.warning(f"{key} is neither block variable nor is it used as a factor in the segment text")
                
                
    # st.write('block_factor_levels', block_factor_levels)
    # st.write('factor_factor_levels', factor_factor_levels)
    
    text_for_llm_sample_list = []
    ranking_display_order = []
    counter = 1

    segment_text = current_round['segment']
    # In Ranking, add all factor levels to the segment
    
    # separate block and factors
    block_factor_dict = dict()
    block_factor_dict_display = dict()
    if block_factor_levels:
        for block_factor_level in block_factor_levels:
            # st.write('block_factor_level', block_factor_level)
            for item in block_factor_level:
                for k, v in item.items():
                    var_name = k.split('|')[0]
                    block_factor_dict[var_name] = item[f'{var_name}|text']
                    block_factor_dict_display[var_name] = item[f'{var_name}|name']
                

    for factor_level in factor_factor_levels:
        # factor_level is a list of dict
        # st.write('factor_level', factor_level)
        fac = {k.split('|')[0]: v for item in factor_level for k, v in item.items() if k.split('|')[1]=='text'}
        fac_display = {k.split('|')[0]: v for item in factor_level for k, v in item.items() if k.split('|')[1]=='name'}
        fac.update(block_factor_dict)
        fac_display.update(block_factor_dict_display)
        
        # st.write('fac_display', fac_display)
        # st.write('fac', fac)
    
        text = segment_text.format(**fac)
        final_text = f"Item {counter}:\n\n{text}"
        counter+=1
        text_for_llm_sample_list.append(final_text)
        ranking_display_order.append(fac_display)         
    # 

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)

    return text_for_llm_sample, ranking_display_order

def get_llm_text_mixed_choice(choice_combination, current_round):
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    filtered_factors_list = filter_factors_list(current_round['factors_list'], round_segment_variables)
    filtered_factors_names = set([ff[0] for ff in filtered_factors_list])
    # st.write(filtered_factors_names)
    
    # st.write('choice_combination', choice_combination)

    block_variables = st.session_state.get('block_variables')

    block_variables_names = set([block_variable[0] for block_variable in block_variables if block_variable])
    # st.write('block_variables_names', block_variables_names)
    block_factor_levels = []
    factor_factor_levels = []
    if block_variables:
        for choice_combo in choice_combination:
            # get factor_level name
            key = set([k.split('|')[0] for fl in choice_combo for k,v in fl.items()])
            # st.write('key', key)
            if key.intersection(block_variables_names):
                block_factor_levels.append(choice_combo)
            elif key.issubset(filtered_factors_names):
                # st.write(f'{key} is subset of {filtered_factors_names}')
                factor_factor_levels.append(choice_combo)
            else:
                st.warning(f"{key} is neither block variable nor is it used as a factor in the segment text")
    
    # st.write('block_factor_levels', block_factor_levels)  
    # st.write('factor_factor_levels', factor_factor_levels)  
    
    text_for_llm_sample_list = []
    choices_display_order = []
    counter = 1   
    
    # separate block and factors
    block_factor_dict = dict()
    block_factor_dict_display = dict()
    if block_factor_levels:
        for block_factor_level in block_factor_levels:
            # st.write('block_factor_level', block_factor_level)
            for item in block_factor_level:
                for k, v in item.items():
                    var_name = k.split('|')[0]
                    block_factor_dict[var_name] = item[f'{var_name}|text']
                    block_factor_dict_display[var_name] = item[f'{var_name}|name']
                
    
    for choice in factor_factor_levels:
        # st.write('choice', choice)
        fac = {k.split('|')[0]: v for item in choice for k, v in item.items() if k.split('|')[1]=='text'}
        fac_display = {k.split('|')[0]: v for item in choice for k, v in item.items() if k.split('|')[1]=='name'}
        fac.update(block_factor_dict)
        fac_display.update(block_factor_dict_display)
        
        # st.write(fac)
        
        # extracted = {k.split('|')[0]:c[k] for c in choice for k in c.keys() if '|text' in k}
        # extracted_name = {k.split('|')[0]:c[k] for c in choice for k in c.keys() if '|name' in k}
        choices_display_order.append(fac_display)
        segment_text_inserted = current_round['segment'].format(**fac)
        text_for_llm_sample_list.append(f'Choice {counter}\n\n{segment_text_inserted}\n\n')
        counter+=1

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample, choices_display_order


def get_llm_text_mixed_scales(factor_product, current_round):
    if not factor_product:
        return '', []
    
    # st.write('factor_product', factor_product)
    
    round_segment_variables = get_segment_text_from_segments_list(current_round)
    
    filtered_factors_list = []
    filtered_factors_names = []
    fp_level_display = dict()
    fp_level = dict()
    for fp in factor_product:
        # st.write('fp', fp)
        fp_name = [k.split('|')[0] for k,v in fp.items()][0]
        fp_level.update({fp_name: fp[f'{fp_name}|text']})
        fp_level_display.update({fp_name: fp[f'{fp_name}|name']})
        # st.write('fp_level', fp_level)
        # st.write('fp_level_display', fp_level_display)
        if fp_name in round_segment_variables:
            filtered_factors_list.append(fp)
            filtered_factors_names.append(fp_name)
    # st.write(filtered_factors_list)
    # st.write(filtered_factors_names)
    # st.write('fp_level', fp_level)
    # st.write('fp_level_display', fp_level_display)
    
    text_for_llm_sample_list = []
    final_text = current_round['segment'].format(**fp_level)
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
    
    text_segment = current_round['segment']
    if not text_segment:
        return []
    variables = re.findall(r"{(.*?)}", text_segment)
    variables = list(set(variables))
    # st.write('variables', variables)
    return variables

def filter_factors_list(factors_list, round_segment_variables):
    return [fac for fac in (factors_list) if fac[0] in round_segment_variables]
