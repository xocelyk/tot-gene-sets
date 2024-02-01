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

test_indices = range(0, 100)

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

def best_of_6_one_example(test_idx, eval_data, model):
    test_gene_set = eval_data[test_idx][1]
    true_label = eval_data[test_idx][2]
    messages = best_of_6_prompt(test_gene_set)
    response = model(messages)
    response = parse_best_of_6_response(response)
    return test_gene_set, response, true_label


MODE = 'best_of_n'

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
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=20, stop='\n')
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
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=20, stop='\n')
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

elif MODE == 'best_of_n':
    save_filename = 'results2/best_of_6_eval.pkl'
    chatgpt = partial(chatgpt, model='gpt-4-1106-preview', temperature=0.7, json=False, max_tokens=1000, stop='\n')
    results = []
    train_data, eval_data = load_few_shot_data()
    num_tries = 6
    for idx in test_indices:
        # choose the answer with the highest similarity score over num_tries
        best_ans = None
        best_score = 0
        best_percentile = 0
        gene_set, candidate_lst, true_label = best_of_6_one_example(idx, eval_data, chatgpt)
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
    df = pd.DataFrame(results)
    df.to_csv('results2/best_of_6_eval.csv', index=False)
    # print summary stats
    print(df.describe())

