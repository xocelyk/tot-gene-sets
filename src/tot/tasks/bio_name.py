import os
import re
from tot.tasks.base import Task, DATA_PATH
# from tot.prompts.bio_name import * #to edit
from tot.prompts.bio_name_kyle import *
from tot.models import gpt
import json
from tot.external_tools.external_tools import *

class Bio_Name(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='toy_sample.txt'):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        xpath = os.path.join(DATA_PATH, 'gene_sets', 'x.txt')
        ypath = os.path.join(DATA_PATH, 'gene_sets', 'y.txt')
        self.data = open(xpath).readlines()
        self.labels = open(ypath).readlines()
        self.steps = 4
        self.stops = ['Continue: No', None] #to edit

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]
    
    def get_label(self, idx: int) -> str:
        return self.labels[idx]
        
    def test_output(self, idx: int, output: str): # to edit
        raise ValueError('no test_output')
#         output = output.split(f'{last_step_prompt}\n')[-1]
#         prompt = score_prompt + output
#         score_outputs = gpt(prompt, n=5, model='gpt-4')
#         scores = []
#         for score_output in score_outputs:
#             # print(score_output)
#             pattern = r".*coherency score is (\d+).*"
#             match = re.match(pattern, score_output, re.DOTALL)
#             if match:
#                 score = int(match.groups()[0])
#                 scores.append(score)
#             else:
#                 print(f'------------------score no match: {[score_output]}')
#         print(scores)
#         # print('------------')
#         info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
#         return info

    def prompt_criticism_wrap(self, x: str, omit_y_path: list[str]) -> str:
        system_message = system_prompt.format(json_format=format_5)
        omit_y_path = ' -> '.join(omit_y_path)
        user_message = criticism_prompt.format(input=x, omit_y_path=omit_y_path, format_5=json.dumps(format_5))
        return system_message, user_message
    
    @staticmethod
    def process_criticism(criticism: str) -> list[str]:
        criticism = criticism.replace('\n','')
        criticism = json.loads(criticism)
        return criticism['Criticism']
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        raise ValueError('no standard_prompt_wrap') #return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        raise ValueError('no cot_prompt_wrap')  #return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='', step_num=None) -> str:
        if step_num == None: raise ValueError('Step not found in the prompt')
        
        if step_num == 0:
            json_format = format_0
            user_message = propose_prompt.format(input=x)
        else:
            json_format = format_1
            user_message = next_step_prompt.format(input=x, step_num=step_num+1, prev_step_num=step_num, y=y, format_1=json.dumps(format_1))
        system_message = system_prompt.format(json_format=json_format)
        return system_message, user_message
    
    @staticmethod
    def propose_prompt_final_wrap(x: str, y: str='') -> str:
        system_message = system_prompt.format(json_format=format_3)
        user_message = last_step_prompt.format(input=x, y=y, format_3=json.dumps(format_3))
        return system_message, user_message
        
    @staticmethod
    def propose_prompt_tools(x: str, y: str='') -> str:
        terms = get_GO_Enrich(x)
        prompt = GO_Enrich_prompt.format(y=y, terms=terms, format_4=json.dumps(format_4))
        return prompt
                    
    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        system_message = system_prompt.format(json_format=format_2)
        choice = ''
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
#             prompt += f'Choice {i}:\n{y}\n'
            choice += f'{y}\n'
        user_message = vote_prompt.format(input=x, choice=choice, format_2=json.dumps(format_2))
        return system_message, user_message
    
    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        raise ValueError('no value_prompt_wrap')
#         last_line = y.strip().split('\n')[-1]
#         if 'left: ' not in last_line:  # last step
#             ans = last_line.lower().replace('answer: ', '')
#             # print([value_last_step_prompt.format(input=x, answer=ans)])
#             return value_last_step_prompt.format(input=x, answer=ans)
#         current_numbers = get_current_numbers(y)
#         return value_prompt.format(input=current_numbers)
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            vote_output = vote_output.replace('\n','')
            vote_output = json.loads(vote_output)
            vote = int(vote_output['Best Biological Process']['index'])
            name = vote_output['Best Biological Process']['Biological Process']
            
            #checking if gpt gives the right index. 
            if vote_output['Analysis'][vote]['Biological Process'] != name:
                for i, y in enumerate(vote_output['Analysis']):
                    if y['Biological  Process'] == name:
                        vote = i
                        break
                        
            if vote in range(n_candidates):
                vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        raise ValueError('no compare_prompt_wrap')
#         assert len(ys) == 2, 'compare prompt only supports 2 candidates'
#         ys = [y.split('Passage:\n')[-1] for y in ys]
#         prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
#         return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        raise ValueError('no compare_output_unwrap')
#         if 'more coherent passage is 1' in compare_output:
#             return 0
#         elif 'more coherent passage is 2' in compare_output:
#             return 1
#         elif 'two passages are similarly coherent' in compare_output:
#             return 0.5
#         else:
#             print(f'-----------------compare no match: {[compare_output]}')
#             return -1
    @staticmethod
    def into_choices(answer: str, y: str, step_num:int):
        # print('Step Num:', step_num)
        # NOTES: changed how we handle the response to the first step.
        # 0th step prompt returns only "Biological Perspective" because there is no need to distinguish between the previous and new biological perspectives.
        # So, no need to change the answer keys before returning the choices
        answer = answer[0]
        answer = answer.replace('\n','')
        answer_f = json.loads(answer)
        answer_f = [answer_f['Answer 1'], answer_f['Answer 2'], answer_f['Answer 3']]
        new_answer = []
        if step_num == 0:
            for a in answer_f:
                new_answer.append(json.dumps(a))
        else:
            for a in answer_f:
                del a['Previous Biological Process']
                a['Biological Process'] = a['New Biological Process']
                del a['New Biological Process']
                new_answer.append(json.dumps(a))
        return new_answer

            
    @staticmethod
    def process_final_answers(answer: str, y: str):
        # print('process_final_answers')
        # print('answer',answer)
        answer = json.loads(answer)
        reasons = answer['Reasons']
        name = answer['Proposed Name']
        # print('name', name)
        # print('reasons',reasons)
        return name, reasons

    
    @staticmethod
    def combine_vote_to_answer(vote_output: str, ys: str):   
        # print('vote_output',vote_output)
        # print('ys',ys)
        vote_output = vote_output.replace('\n','')
        vote_output = json.loads(vote_output)
        
        for i, (v, _) in enumerate(zip(vote_output['Analysis'], ys)):
            ys[i] = json.loads(ys[i])
            if v['Biological Process'] == ys[i]["Biological Process"]:
                ys[i].update({'Pros': v['Pros']})
                ys[i].update({'Cons': v['Cons']})
            # TODO: why would they be out of order?
            else:
                for j, v_ in enumerate(vote_output['Analysis']):
                    if v_['Biological Process'] == ys[i]["Biological Process"]:
                        ys[j].update({'Pros': v['Pros']})
                        ys[j].update({'Cons': v['Cons']})
                        break
        
        # print('ys',ys)
        return [json.dumps(y) for y in ys]
    
    @staticmethod
    def combine_tools_to_answer(tool_output: str, ys: str):   

        tool_output = tool_output.replace('\n','')
        tool_output = json.loads(tool_output)
        
        for i, (v, _) in enumerate(zip(tool_output['Analysis'], ys)):
            ys[i] = json.loads(ys[i])
            if v['Biological Process'] == ys[i]["Biological Process"]:
                ys[i].update({'Comparison': v['Comparison']})
            else:
                for j, v_ in enumerate(tool_output['Analysis']):
                    if v_['Biological Process'] == ys[i]["Biological Process"]:
                        ys[j].update({'Comparison': v['Comparison']})
                        break
        return [json.dumps(y) for y in ys]
    
