import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import json


def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_votes_for_bionames(task, x, ys, n_evaluate_sample, step):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    # print('get_votes_for_bionames -- vote_prompt',vote_prompt)
    vote_outputs = gpt(vote_prompt, n=n_evaluate_sample, stop=None)
    # print('get_votes_for_bionames -- vote_outputs',vote_outputs)
    #vote_outputs = task.process_vote_outputs(vote_output) #to do, summarize votes
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values, vote_outputs

def combine_vote_to_answer(task, vote_outputs, ys):
    new_ys = task.combine_vote_to_answer(vote_outputs[0], ys)
    return new_ys

def get_proposals(task, x, y): 
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]

def get_samples_for_bionames(task, x, y, n_generate_sample, prompt_sample, step): 
    system_message, user_message = task.propose_prompt_wrap(x, y, step)
#     print('get_samples_for_bionames -- step',step)
#     print('get_samples_for_bionames -- x',x)
    # print('get_samples_for_bionames -- propose_prompt', propose_prompt)
    samples = gpt(system_message, user_message, n=1)
    print(samples)
    # print('get_samples_for_bionames -- samples', samples)
    samples = task.into_choices(samples[0], y, step)
#     print('get_samples_for_bionames -- processed_answers', samples)
    return samples

def get_tool_reflection(task, x, ys): 
    propose_prompt = task.propose_prompt_tools(x,ys)
    # print('get_tool_reflection -- propose_prompt', propose_prompt)
    tools_output = gpt(propose_prompt, n=1)
    # print('get_tool_reflection -- tools_output',tools_output)
    ys = task.combine_tools_to_answer(tools_output[0], ys)
    return ys

def get_final_answer_for_bionames(task, x, y, n_generate_sample, prompt_sample): 
    propose_prompt = task.propose_prompt_final_wrap(x, y)
#     print('get_final_answer_for_bionames -- final_propose_prompt', propose_prompt)
    samples = gpt(propose_prompt, n=1)
#     print('get_final_answer_for_bionames -- final_samples', samples)
    final_answer, samples = task.process_final_answers(samples[0], y)
#     print('get_final_answer_for_bionames -- final processed_answers', samples)
#     print('get_final_answer_for_bionames -- final_answer',final_answer)
    return final_answer, samples

def get_criticism_for_bionames(task, x, omit_y_path):
    propose_prompt = task.prompt_criticism_wrap(x, omit_y_path)
    # print('get_criticisms_for_bionames -- propose_prompt', propose_prompt)
    sample = gpt(propose_prompt, n=1)
    # print('get_criticisms_for_bionames -- samples', samples)
    sample = task.process_criticism(sample[0])
    sample = {'Proteins': x, 'Path': omit_y_path, 'Criticism': sample}
    # print('get_criticisms_for_bionames -- processed_answers', samples)
    return sample

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    max_mem_size = 3
    mem = []  # cache for self-reflection
    y_paths = [[]]  # current output candidate paths
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # update memory to keep under max_mem_size
        mem = mem[-max_mem_size:]

        new_y_paths = []
        # generation
        if args.method_generate == 'sample':
            new_ys = [get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
            
        elif args.method_generate == 'sample_bionames':
            new_ys = []
            for i, y in enumerate(ys):
                item = get_samples_for_bionames(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, step=step)
                new_ys.append(item)

                for y in item:
                    new_y_paths.append(y_paths[i] + [y])
                
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys] 
            
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'votes_for_bionames':
            values, vote_outputs = get_votes_for_bionames(task, x, new_ys, args.n_evaluate_sample, step)
            #combine pros and cons to each perspective
            new_ys = combine_vote_to_answer(task, vote_outputs, new_ys)
            # new_ys = get_tool_reflection(task, x, new_ys)
        
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]

        y_paths = [new_y_paths[select_id] for select_id in select_ids]
        omit_y_paths = [new_y_paths[select_id] for select_id in ids if select_id not in select_ids]
        omit_y_paths = [[json.loads(y)['Biological Perspective'] for y in omit_y_path] for omit_y_path in omit_y_paths]
        criticisms = []
        for omit_y_path in omit_y_paths:
            criticisms.append(get_criticism_for_bionames(task, x, omit_y_path))
        mem.extend(criticisms)
        print('solve -- criticisms', criticisms)
        
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if len(omit_y_paths) > 0:
            print('omitted paths', omit_y_paths)
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
  
    if args.task == 'bio_name':
        final_answer, new_ys = get_final_answer_for_bionames(task, x, ys[0], args.n_generate_sample, prompt_sample=args.prompt_sample)
        infos.append({'step': step+1, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': None, 'select_new_ys': None})
    if to_print: 
        print('solve -- ys', ys)
    
    return final_answer, ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}