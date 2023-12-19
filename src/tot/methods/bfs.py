import itertools
import numpy as np
from functools import partial
from tot.models import gpt
import json
from graphviz import Digraph

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
    values = task.vote_outputs_unwrap(vote_outputs, ys)
    return values

def get_votes_for_bionames(task, x, ys, n_evaluate_sample, step):
    system_message, user_message  = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(system_message, user_message, n=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, ys)
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
    samples = gpt(prompt, n=1, stop=stop)
    return [y + _ for _ in samples]

def get_samples_for_bionames(task, x, y, n_generate_sample, prompt_sample, step):
    system_message, user_message = task.propose_prompt_wrap(x, y, step)
    samples = gpt(system_message, user_message, n=1)
    samples = task.into_choices(samples, y, step)
    return samples

def get_tool_reflection(task, x, ys): 
    propose_prompt = task.propose_prompt_tools(x,ys)
    tools_output = gpt(propose_prompt, n=1)
    ys = task.combine_tools_to_answer(tools_output[0], ys)
    return ys

def get_final_answer_for_bionames(task, x, y, n_generate_sample, prompt_sample): 
    system_message, user_message = task.propose_prompt_final_wrap(x, y)
    samples = gpt(system_message, user_message, n=1)
    final_answer, samples = task.process_final_answers(samples[0], y)
    return final_answer, samples

def get_criticism_for_bionames(task, x, omit_y_path):
    # TODO: fix
    system_message, user_message = task.prompt_criticism_wrap(x, omit_y_path)
    sample = gpt(system_message, user_message, n=1)
    sample = task.process_criticism(sample[0])
    sample = {'Genes': x, 'Path': omit_y_path, 'Criticism': sample}
    return sample

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None
    
    def insert(self, word, value):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = value

    def search(self, word):
        node = self
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.value
    
    def visualize(self, graph, parent_name, char, end):
        # Create a unique name for each node based on its character and parent
        node_name = f"{parent_name}{char}_{id(self)}"
        if end:
            graph.node(node_name, f'{char}', shape='circle')
        else:
            graph.node(node_name, char)
        
        # Connect this node to its parent in the graph
        if parent_name:
            graph.edge(parent_name, node_name)
        
        # Recursively visualize children nodes
        for child_char, child_node in self.children.items():
            child_node.visualize(graph, node_name, child_char + ' ' + '({})'.format(child_node.value), child_node.is_end)

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word, value):
        self.root.insert(word, value)
    
    def search(self, word):
        return self.root.search(word)
    
    def visualize(self):
        graph = Digraph(comment='Trie')
        if self.root:
            self.root.visualize(graph, '', 'Biological Process', False)
        return graph

    def print_tree(self):
        print(self.root.value)
        for child in self.root.children:
            print(child.value)
            for grandchild in child.children:
                print(grandchild.value)


def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    max_mem_size = 3
    trie = Trie()
    
    mem = []  # cache for self-reflection
    y_paths = [[]]  # current output candidate paths
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        print('-- step', step, '--')
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
                # new_ys.append(item)

                for y in item:
                    new_ys.append(y)
                    new_y_paths.append(y_paths[i] + [y])
                
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys] 
            
        # new_ys = list(itertools.chain(*new_ys))
        # print('-- new ys pre vote --')
        # for y in new_ys:
        #     print(json.loads(y)['Biological Process'])
        # there is only one output each time rn
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'votes_for_bionames':
            values, vote_outputs = get_votes_for_bionames(task, x, new_ys, args.n_evaluate_sample, step)
            # print(' -- vote outputs --')
            # print(vote_outputs)
            # print(' -- values --')
            # print(values)
            #combine pros and cons to each perspective
            # new_ys = combine_vote_to_answer(task, vote_outputs, new_ys)
            # new_ys = get_tool_reflection(task, x, new_ys)
        
        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
            print(' -- select ids --')
            print(select_ids)

        select_new_y_paths = [new_y_paths[select_id] for select_id in select_ids]
        omit_y_paths = [new_y_paths[id] for id in ids if id not in select_ids]
        omit_y_paths = [[json.loads(y)['Biological Process'] for y in omit_y_path] for omit_y_path in omit_y_paths]

        print('-- paths --')
        for i in range(len(new_y_paths)):
            path = [json.loads(y)['Biological Process'] for y in new_y_paths[i]]
            print(' -> '.join(path))
            value = values[i]
            if trie.search(path) is None:
                trie.insert(path, value)
            else:
                trie.insert(path, max(value, trie.search(path)))

        # criticisms = []
        # for omit_y_path in omit_y_paths:
        #     criticisms.append(get_criticism_for_bionames(task, x, omit_y_path))
        # mem.extend(criticisms)
        # print('solve -- criticisms', criticisms)
        
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        if len(omit_y_paths) > 0:
            pass
            # print('omitted paths', omit_y_paths)
        # log
        if to_print: 
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            # bp_y_paths = [[json.loads(y)['Biological Process'] for y in y_path] for y_path in new_y_paths]
            print(f'-- new_ys --: {[json.loads(y)["Biological Process"] for y in sorted_new_ys]}\n-- sol values --: {sorted_values}\n-- choices --: {[json.loads(y)["Biological Process"] for y in select_new_ys]}\n')
            # print('-- y paths --: {}\n'.format([' -> '.join(path) for path in bp_y_paths]))
        
        infos.append({'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys
        y_paths = select_new_y_paths

        dot = trie.visualize()
        dot.render('viz/trie_visualization_{}'.format(idx), format='png')
  
    if args.task == 'bio_name':
        final_answer, new_ys = get_final_answer_for_bionames(task, x, ys[0], args.n_generate_sample, prompt_sample=args.prompt_sample)
        infos.append({'step': step+1, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': None, 'select_new_ys': None})
    if to_print: 
        pass
        # print('solve -- ys', ys)

    dot = trie.visualize()
    dot.render('viz/trie_visualization_'.format(idx), format='png')
    for path in y_paths:
        if [json.loads(y)['Biological Process'] for y in path][-1] == final_answer:
            final_path = path
            break
    
    trie.insert(final_path, '*')
    
    return final_answer, ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}