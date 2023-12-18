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
        json_format = format_0
        system_message = system_prompt.format(json_format = json_format)
        if step_num == 0:
            user_message = propose_prompt.format(input=x, format_0=json.dumps(format_0)) + y
        else:
            user_message = next_step_prompt.format(input=x, step_num=step_num+1, prev_step_num=step_num, y=y, format_1=json.dumps(format_1))
        
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
            vote = int(vote_output['Best biological perspective']['index'])
            name = vote_output['Best biological perspective']['Biological Perspective']
            
            #checking if gpt gives the right index. 
            if vote_output['Analysis'][vote]['Biological Perspective'] != name:
                print('vote_outputs_unwrap, wrong index')
                for i, y in enumerate(vote_output['Analysis']):
                    if y['Biological  Perspective'] == name:
                        vote = i
                        break
                        
            if vote in range(n_candidates):
                vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
            
#             pattern = r".*best biological perspective is .*(\d+).*"
#             match = re.match(pattern, vote_output, re.DOTALL)
#             if match:
#                 vote = int(match.groups()[0]) - 1
#                 if vote in range(n_candidates):
#                     vote_results[vote] += 1
#             else:
#                 print(f'vote no match: {[vote_output]}')
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
        print('Step Num:', step_num)
        # NOTES: changed how we handle the response to the first step.
        # 0th step prompt returns only "Biological Perspective" because there is no need to distinguish between the previous and new biological perspectives.
        # So, no need to change the answer keys before returning the choices
        answer = answer.replace('\n','')
        answer_f =  json.loads(answer)
        print('Answers:')
        for answer_choice in answer_f:
            if step_num == 0:
                print(answer_choice['Biological Perspective'])
            else:
                print(answer_choice['New Biological Perspective'])
        new_answer = []
        if step_num == 0:
            for a in answer_f:
                new_answer.append(json.dumps(a))
        else:
            for a in answer_f:
                del a['Selected Biological Perspective']
                a['Biological Perspective'] = a['New Biological Perspective']
                del a['New Biological Perspective']
                new_answer.append(json.dumps(a))
#         y_f =  json.loads(y)
        
#         for a in answer_f:
#             y_f.append(a)
        
        return new_answer
        
#         pattern = re.compile(r'(Biological Perspective \d(?:_\d)*:.*?)(?=Biological Perspective \d(?:_\d)*:|$)', re.DOTALL)
#         matches = pattern.findall(answer)

#         if step_num > 0:
#             outputs = []
#             processed_outputs = []
#             for match in matches:
#                 pattern_2 = re.compile(r'Biological Perspective (\d(?:_\d)*)')
#                 num1 = int(pattern_2.findall(match)[0][0]) #number of perspectives: e.g., Biological Perspective 2_1 --> 2
#                 num2 = int(pattern_2.findall(y)[0]) #number of the perspective from step 1
#                 if num1 == num2:
#                     outputs.append(f'{y}\n{match}')
#                     processed_outputs.append(f'{y}\n Step:{step_num+1}\n{match}')
#         else:
#             outputs = [_ + '\n' for _ in matches]
#             processed_outputs = [f'\nStep: {step_num+1}\n' + _ + '\n' for _ in matches]
#         return outputs, processed_outputs
            
    @staticmethod
    def process_final_answers(answer: str, y: str):
        print('process_final_answers')
        print('answer',answer)
        answer = json.loads(answer)
        reasons = answer['Reasons']
        name = answer['Proposed Name']
        print('name', name)
        print('reasons',reasons)
        return name, reasons
#         pattern = re.compile(r'Proposed Name: (.+)')
#         match = pattern.search(answer)
#         if match:
#             final_answer = match.group(1).strip()
#         else:
#             final_answer = None
#             print(f'Proposed Name no match')
#         return final_answer, y + answer
    
#     @staticmethod
#     def process_vote_outputs(vote_outputs: list):
  
#         return new_ys
    
    @staticmethod
    def combine_vote_to_answer(vote_output: str, ys: str):   
        # print('vote_output',vote_output)
        # print('ys',ys)
        vote_output = vote_output.replace('\n','')
        vote_output = json.loads(vote_output)
        
        for i, (v, _) in enumerate(zip(vote_output['Analysis'], ys)):
            ys[i] = json.loads(ys[i])
            if v['Biological Perspective'] == ys[i]["Biological Perspective"]:
                ys[i].update({'Pros': v['Pros']})
                ys[i].update({'Cons': v['Cons']})
            # TODO: why would they be out of order?
            else:
                for j, v_ in enumerate(vote_output['Analysis']):
                    if v_['Biological Perspective'] == ys[i]["Biological Perspective"]:
                        ys[j].update({'Pros': v['Pros']})
                        ys[j].update({'Cons': v['Cons']})
                        break
        
        print('ys',ys)
        return [json.dumps(y) for y in ys]
    
    @staticmethod
    def combine_tools_to_answer(tool_output: str, ys: str):   

        tool_output = tool_output.replace('\n','')
        tool_output = json.loads(tool_output)
        
        for i, (v, _) in enumerate(zip(tool_output['Analysis'], ys)):
            ys[i] = json.loads(ys[i])
            if v['Biological Perspective'] == ys[i]["Biological Perspective"]:
                ys[i].update({'Comparison': v['Comparison']})
            else:
                for j, v_ in enumerate(tool_output['Analysis']):
                    if v_['Biological Perspective'] == ys[i]["Biological Perspective"]:
                        ys[j].update({'Comparison': v['Comparison']})
                        break
        return [json.dumps(y) for y in ys]
        
        
        
        
#         for match, y in zip(matches, ys):
            
#         pattern = r'(?P<title>Biological Perspective) (?P<number>\d[^\n]*)(?:: [^-\n]+)?\n(?:- )?(?P<pros>Pros:[^\n]+)\n(?:- )?(?P<cons>Cons:[^\n]+)'

#         matches = re.finditer(pattern, text)
#         new_ys = []
#         # combine pros/cons to ys. 
#         for match, y in zip(matches, ys):
#             label = match.group('title')
#             number = match.group('number')[0]
#             pros = match.group('pros')
#             cons = match.group('cons')

#             pattern = re.compile(r'Biological Perspective (\d(?:_\d)*)')
#             matches = pattern.findall(y)
#             y_num = matches[0]
#             combined_content = f"Pros & Cons:\n{pros}\n{cons}"
#             new_ys.append(f'{y}{combined_content}')
           
    
    
    
          #---
#         # Use regex to match each block of text starting with "Biological Perspective" 
#         matches = re.findall(r'(Biological Perspective) (\d+):[^\n]+(?:\n- (Pros:[^\n]+))(?:\n- (Cons:[^\n]+))', text)

#         assert len(matches) == len(ys), f'combine_vote_to_answer -- {len(matches)} != {len(ys)} \n text\n{text}'
        
#         new_ys = []
#         # combine pros/cons to ys. 
#         for match, y in zip(matches, ys):
#             label, number, pros, cons = match
#             pattern = re.compile(r'Biological Perspective (\d(?:_\d)*)')
#             matches = pattern.findall(y)
#             y_num = matches[0]
#             if number != y_num:
#                 print('combine_vote_to_answer -- match', match)
#                 print('combine_vote_to_answer -- y',y)
#                 print('combine_vote_to_answer -- number',number)
#                 print('combine_vote_to_answer -- y_num',y_num)
#                 error
#             combined_content = f"Pros & Cons:\n{pros}\n{cons}"
#             print(f'combine_vote_to_answer -- Step: {step_num+1}\n{y}\n{combined_content}\n')
#             new_ys.append(f'Step: {step_num+1}\n{y}\n{combined_content}')
                
#         return new_ys
    
