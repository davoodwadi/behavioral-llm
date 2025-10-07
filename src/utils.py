import streamlit as st
import random 

def get_llm_text_mixed_rank(current_round, factor_levels, block_variable_level=None, block_variable_name=None):
    segments = current_round['segments']
    text_for_llm_sample_list = []
    ranking_display_order = []
    counter = 1  
    for seg in segments:
        if 'Fixed' in seg['segment_label']:
            text_for_llm_sample_list.append(seg['segment_text']) 
        elif 'Ranking' in seg['segment_label']:
            segment_text = seg['segment_text']
            # In Ranking, for each segment text, add all factor levels
            for factor_level in factor_levels:
                factor_name = [k.split('|')[0] for k, v in factor_level.items()][0]
                fac = {factor_name:factor_level[f'{factor_name}|text']}
            
                if block_variable_name:
                    fac[block_variable_name] = block_variable_level[f'{block_variable_name}|text']
            
                text = segment_text.format(**fac)
                final_text = f"Item {counter}:\n\n{text}"
                counter+=1
                text_for_llm_sample_list.append(final_text)
                ranking_display_order.append(factor_level[f'{factor_name}|name'])         
            # 

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)

    return text_for_llm_sample, ranking_display_order

def get_llm_text_mixed_choice(choice_combination, current_round):
    text_for_llm_sample_list = []
    choices_display_order = []
    counter = 1   
    for segment in current_round['segments']:
        if 'Choice' in segment['segment_label']:
            for choice in choice_combination:
                extracted = {k.split('|')[0]:c[k] for c in choice for k in c.keys() if '|text' in k}
                extracted_name = {k.split('|')[0]:c[k] for c in choice for k in c.keys() if '|name' in k}
                choices_display_order.append(extracted_name)
                segment_text_inserted = segment['segment_text'].format(**extracted)
                text_for_llm_sample_list.append(f'Choice {counter}\n\n{segment_text_inserted}\n\n')
                counter+=1
        else:
            text_for_llm_sample_list.append(segment.get('segment_text'))
    # st.write(choices_display_order)
    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample, choices_display_order


def get_llm_text_mixed_scales(factor_product, current_round):
    # st.write(factor_product)
    if not factor_product:
        return '', []
    
    d = {k.split('|')[0]:v for factor in factor_product for k,v in factor.items() if 'text' in k}
    factor_display = {k.split('|')[0]:v for factor in factor_product for k,v in factor.items() if 'name' in k}
    # st.write('d', d)
    text_for_llm_sample_list = []
    for segment in current_round['segments']:
        if 'Treatment' in segment['segment_label']:
            final_text = segment['segment_text'].format(**d)
            text_for_llm_sample_list.append(final_text)

        else:
            text_for_llm_sample_list.append(segment['segment_text'])

    text_for_llm_sample = '\n\n---\n\n'.join(text_for_llm_sample_list)
    return text_for_llm_sample, factor_display

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