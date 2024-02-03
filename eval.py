import sys
sys.path.insert(0, 'src') 
import os
import json
from functools import partial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import pickle
from tqdm import tqdm
import argparse
from tot.methods.bfs import solve
from tot.tasks.bio_name import Bio_Name
from utils import *
from tot.models import *

FLAG = True
test_indices = range(0, 100)
MODE = 'zero_shot_alt'
print(MODE)

def preload(filename):
    # assume filename is a pickle file
    return pickle.load(open(filename, 'rb'))

def eval(task, args, save_filename, start_idx=0, stop_idx=100, preload=False):
    if preload:
        results = pickle.load(open(save_filename, 'rb'))
    else:
        results = []
    eval_indices = range(start_idx, stop_idx)
    for idx in tqdm(eval_indices):
        results.append(test_example_wrap(idx, args, task))
        with open(save_filename, 'wb') as f:
            pickle.dump(results, f)

def few_shot_one_example(test_idx, eval_data, train_data, model, n=5):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    indices = np.random.choice(len(train_data), n, replace=False)
    genes = [train_data[i][1] for i in indices]
    labels = [train_data[i][2] for i in indices]
    messages = get_few_shot_prompt(genes, labels, test_gene_set)
    response = model(messages)
    response = parse_few_shot_response(response)
    return test_gene_set, response, true_label

def zero_shot_one_example(test_idx, eval_data, model):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    messages = get_zero_shot_prompt(test_gene_set)
    response = model(messages)
    response = parse_zero_shot_response(response)
    return test_gene_set, response, true_label

def zero_shot_alt_one_example(test_idx, eval_data, model):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    messages = get_zero_shot_prompt_alt(test_gene_set)
    response = model(messages)
    response = parse_zero_shot_response(response)
    return test_gene_set, response, true_label

def best_of_9_one_example(test_idx, eval_data, model):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    messages = best_of_9_prompt(test_gene_set)
    response = model(messages)
    response = parse_best_of_n_response(response)
    return test_gene_set, response, true_label

def best_of_27_one_example(test_idx, eval_data, model):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    messages = best_of_27_prompt(test_gene_set)
    response = model(messages)
    response = parse_best_of_n_response(response)
    return test_gene_set, response, true_label

if MODE == 'tot':
    args = argparse.Namespace(backend='gpt-4-1106-preview', temperature=0.7, task='bio_name', naive_run=False, prompt_sample=None, method_generate='sample_bionames', method_evaluate='votes_for_bionames', method_select='greedy', n_generate_sample=3, n_evaluate_sample=2, n_select_sample=2)
    task = Bio_Name()
    save_filename = 'results2/tot_eval.pkl'
    # TODO: test indices
    eval(task, args, save_filename, start_idx=min(test_indices), stop_idx=max(test_indices), preload=False)

elif MODE == 'tot_profiler':
    args = argparse.Namespace(backend='gpt-4-1106-preview', temperature=0.7, task='bio_name', naive_run=False, \
                          prompt_sample=None, method_generate='sample_bionames', \
                          method_evaluate='multi_voters', method_select='greedy', n_generate_sample=3, \
                          n_evaluate_sample=2, n_select_sample=2, \
                          source='GO:BP', bio_type="Biological Process",filter_method='sim', filter_size=5, voting_setting=None,\
                         )      
    task = Bio_Name()
    save_filename = 'results2/eval_gprofiler_eval.pkl'
    eval(task, args, save_filename, start_idx=min(test_indices), stop_idx=max(test_indices), preload=True)

elif MODE == 'few_shot':
    save_filename = 'few_shot_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    for idx in test_indices:
        test_gene_set, response, true_label = few_shot_one_example(idx, eval_data, train_data, chatgpt, n=5)
        similarity_score = get_similarity_score(response, true_label)
        similarity_percentile = get_similarity_percentile(similarity_score, response)
        example_result = {'index': idx, 'x': test_gene_set, 'y_pred': response, 'y_true': true_label, 'similarity_score': similarity_score, 'similarity_percentile': similarity_percentile}
        print(example_result)
        results.append(example_result)
        with open(save_filename, 'wb') as f:
            pickle.dump(results, f)
    # turn to dataframe and save as csv
    df = pd.DataFrame(results)
    df.to_csv('results2/few_shot_eval.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'zero_shot':
    save_filename = 'results2/zero_shot_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    for idx in test_indices:
        gene_set, response, true_label = zero_shot_one_example(idx)
        similarity_score = get_similarity_score(response, true_label)
        similarity_percentile = get_similarity_percentile(similarity_score, response)
        example_result = {'Index': idx, 'x': gene_set, 'y_pred': response, 'y_true': true_label, 'similarity_score': similarity_score, 'similarity_percentile': similarity_percentile}
        print(example_result)
        results.append(example_result)
        pickle.dump(results, open(save_filename, 'wb'))
    # turn to dataframe and save as csv
    df = pd.DataFrame(results)
    df.to_csv('results2/zero_shot_eval.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'zero_shot_alt':
    save_filename = 'results2/zero_shot_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    for idx in test_indices:
        gene_set, response, true_label = zero_shot_alt_one_example(idx, eval_data, chatgpt)
        similarity_score = get_similarity_score(response, true_label)
        similarity_percentile = get_similarity_percentile(similarity_score, response)
        example_result = {'Index': idx, 'x': gene_set, 'y_pred': response, 'y_true': true_label, 'similarity_score': similarity_score, 'similarity_percentile': similarity_percentile}
        print(example_result)
        results.append(example_result)
        pickle.dump(results, open(save_filename, 'wb'))
    # turn to dataframe and save as csv
    df = pd.DataFrame(results)
    df.to_csv('results2/zero_shot_eval_alt.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'best_of_9':
    save_filename = 'results2/best_of_9_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    num_tries = 9
    for idx in test_indices:
        # choose the answer with the highest similarity score over num_tries
        best_ans = None
        best_score = 0
        best_percentile = 0
        gene_set, candidate_lst, true_label = best_of_9_one_example(idx, eval_data, chatgpt)
        for response in candidate_lst:
            similarity_score = get_similarity_score(response, true_label)
            similarity_percentile = get_similarity_percentile(similarity_score, response)
            # print(response, similarity_score, similarity_percentile)
            if similarity_score > best_score:
                best_ans = response
                best_score = similarity_score
                best_percentile = similarity_percentile
        example_result = {'Index': idx, 'x': gene_set, 'y_pred': best_ans, 'y_true': true_label, 'similarity_score': best_score, 'similarity_percentile': best_percentile}
        results.append(example_result)
        print(example_result)
        pickle.dump(results, open(save_filename, 'wb'))
    # turn to dataframe and save as csv
    df = pd.DataFrame(results, index=False)
    df.to_csv('results2/best_of_9.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'best_of_27':
    save_filename = 'results2/best_of_27_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    num_tries = 27
    for idx in test_indices:
        # choose the answer with the highest similarity score over num_tries
        best_ans = None
        best_score = 0
        best_percentile = 0
        gene_set, candidate_lst, true_label = best_of_27_one_example(idx, eval_data, chatgpt)
        for response in candidate_lst:
            similarity_score = get_similarity_score(response, true_label)
            similarity_percentile = get_similarity_percentile(similarity_score, response)
            # print(response, similarity_score, similarity_percentile)
            if similarity_score > best_score:
                best_ans = response
                best_score = similarity_score
                best_percentile = similarity_percentile
        example_result = {'Index': idx, 'x': gene_set, 'y_pred': best_ans, 'y_true': true_label, 'similarity_score': best_score, 'similarity_percentile': best_percentile}
        results.append(example_result)
        print(example_result)
        pickle.dump(results, open(save_filename, 'wb'))
    # turn to dataframe and save as csv
    df = pd.DataFrame(results, index=False)
    df.to_csv('results2/best_of_27.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'avg_of_9':
    save_filename = 'results2/avg_of_9_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    num_tries = 9
    for idx in test_indices:
        # choose the answer with the highest similarity score over num_tries
        all_responses = []
        all_scores = []
        all_percentiles = []
        gene_set, candidate_lst, true_label = best_of_9_one_example(idx, eval_data, chatgpt)
        for response in candidate_lst:
            similarity_score = get_similarity_score(response, true_label)
            similarity_percentile = get_similarity_percentile(similarity_score, response)
            all_responses.append(response)
            all_scores.append(similarity_score)
            all_percentiles.append(similarity_percentile)
        avg_score = np.mean(all_scores)
        avg_percentile = np.mean(all_percentiles)
        example_result = {'Index': idx, 'x': gene_set, 'y_true': true_label, 'similarity_score': avg_score, 'similarity_percentile': avg_percentile}
        results.append(example_result)
        print(example_result)
        pickle.dump(results, open(save_filename, 'wb'))
    # turn to dataframe and save as csv
    df = pd.DataFrame(results)
    df.to_csv('results2/avg_of_9.csv', index=False)
    # print summary stats
    print(df.describe())

elif MODE == 'hue_et_al':
    train_data, eval_data = load_few_shot_data()
    save_filename = 'results2/hue_et_al.pkl'
    def add_gene_feature_summary(prompt_text, feature_dataframe, n_genes=2):
        for index, row in feature_dataframe.iterrows():
            number_of_genes = 0
            if row['Number of Genes'] is not None:
                number_of_genes = int(row['Number of Genes'])
            if number_of_genes >= n_genes:
                prompt_text += f"{row['Feature']}: {row['Number of Genes']} proteins: {row['Genes']}\n"
        return prompt_text
    
    def make_expert_prompt(genes, feature_df = False, direct = False, customized_prompt = None):
        """
        Create a ChatGPT prompt based on the list of genes
        :return: A string containing the ChatGPT prompt text
        """

        general_analysis_instructions = """
    Be concise, do not use unneccesary words. Be specific, avoid overly general
    statements such as 'the proteins are involved in various cellular processes'
    Be factual, do not editorialize.
    For each important point, describe your reasoning and supporting information.
        """

        task_instructions = """
    Write a critical analysis of the biological processes performed
    by this system of interacting proteins. Propose a brief name for
    for the most prominant biological process performed by the system.
        """
        direct_instructions = """
    Propose a name and provide analysis for the following gene set.
        """

        format_placeholder = """
    Put the name at the top of the analysis as 'Process: <name>'
        """

        if direct == True:
            prompt_text = direct_instructions
            prompt_text += format_placeholder
        elif customized_prompt:
            prompt_text = customized_prompt
            prompt_text += format_placeholder
        else:
            prompt_text = task_instructions
            prompt_text += format_placeholder
            prompt_text += general_analysis_instructions

        prompt_text += "\n\nHere are the interacting proteins:\n"
        prompt_text += f'\nProteins: '
        prompt_text += ", ".join(genes) + ".\n\n"

        if feature_df:
            prompt_text += "\n\nHere are the gene features:\n"
            prompt_text  = add_gene_feature_summary(prompt_text, feature_df)
        return prompt_text

    def hue_et_al_one_example(test_idx, eval_data, model, feature_df, direct = False, customized_prompt = None):
        test_gene_set = eval_data[test_idx][1]
        test_gene_set = test_gene_set.split(' ')
        true_label = eval_data[test_idx][2]
        prompt = make_expert_prompt(test_gene_set, feature_df, direct, customized_prompt)
        global FLAG
        if FLAG:
            print(prompt)
            FLAG = False
        messages = [{'role': 'user', 'content': prompt}]
        response = model(messages)
        response = parse_hue_et_al(response)
        return test_gene_set, response, true_label
    
    def parse_hue_et_al(response):
        response = response[0]
        response = response.split('Process: ')[1].strip()
        return response


    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')

    results = []
    for idx in test_indices:
        test_gene_set, response, true_label = hue_et_al_one_example(idx, eval_data, chatgpt, feature_df=False, direct = False, customized_prompt = None)
        similarity_score = get_similarity_score(response, true_label)
        similarity_percentile = get_similarity_percentile(similarity_score, response)
        example_result = {'index': idx, 'x': test_gene_set, 'y_pred': response, 'y_true': true_label, 'similarity_score': similarity_score, 'similarity_percentile': similarity_percentile}
        print(example_result)
        results.append(example_result)
        with open(save_filename, 'wb') as f:
            pickle.dump(results, f)
    # turn to dataframe and save as csv
    df = pd.DataFrame(results)
    df.to_csv('results2/hue_et_al.csv', index=False)
    # print summary stats
    print(df.describe())

else:
    raise ValueError(f"Invalid mode: {MODE}")


