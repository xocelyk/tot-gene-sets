# TODO: is there extra whitespace surrounding the input? proteins or genes?
# TODO: I think sometimes it is doing step num 1, 2, 3 for different items in same step

# CURRENTLY: generate three reasons for each step

system_prompt = 'You are a helpful and knowledgable assistant to a molecular biologist. Respond to questions in JSON format, following this template: {json_format}.'

# propose initial biological processes
format_0 = {"Answer 1": {"Step": "1", "Biological Process": "<Your first proposed biological process>", "Reason": "<Why did you choose this name?>"},           "Answer 2": {"Step": "1", "Biological Process": "<Your second proposed biological process>", "Reason": "<Why did you choose this name?>"},
            "Answer 3": {"Step": "1", "Biological Process": "<Your third proposed biological process>", "Reason": "<Why did you choose this name?>"}}

# propose more specific biological processes
format_1 = {"Answer 1": {"Step": "{step_num}",\
           "Previous Biological Process": "<The previous biological process>",\
           "New Biological Process": "<Your first proposed biological process, more specific than the previous",\
            "Reason": "<Why did you choose this name?>"},\
            "Answer 2": {"Step": "{step_num}",\
           "Previous Biological Process": "<The previous biological process>",\
           "New Biological Process": "<Your second proposed biological process, more specific than the previous",\
            "Reason": "<Why did you choose this name?>"},\
            "Answer 3": {"Step": "{step_num}",\
           "Previous Biological Process": "<The previous biological process>",\
           "New Biological Process": "<Your third proposed biological process, more specific than the previous",\
            "Reason": "<Why did you choose this name?>"}}


# vote prompt
format_2 = {'Votes': [{'Biological Process': '<The First biological process>'}, {'Biological Process': '<The Second biological process>'}]}

# last step prompt
format_3 = {'Biological Process': '<The best biological process>', 'Reason': '<Your reasoning>'}

propose_prompt = '''
You are given a set of genes, and your task is to propose three high-level biological processes that may be likely to be performed by the system involving expression of these genes. Biological processes are organized in a hierarchical ontology, and the most general biological processes are at the top of the hierarchy. For example, "cellular process" is a general biological process, and "cellular response to stimulus" is a more specific biological process. You should propose three biological processes that are as general as possible.

Here is the set of genes:
Genes: {input}

Given the set of genes, propose three biological processes that may be likely to be performed by the system involving expression of these genes.
'''

next_step_prompt = '''
Given a set of genes and proposed biological processes describing the system, your task is to generate more specific biological processes describing the system. Biological processes are organized in a hierarchical ontology, and the most general biological processes are at the top of the hierarchy. For example, "cellular process" is a general biological process, and "cellular response to stimulus" is a more specific biological process. You should propose three biological processes that are more specific than the proposed biological process.

Here is the set of genes:
Genes: {input}

Here is the current biological process:
---
{y}
---

Given the set of genes and the current biological process, propose three biological processes that are more specific than the current biological process.
'''

last_step_prompt = '''
Given a set of genes and proposed biological processes describing the system, your task is to choose the most accurate biological process.

Genes: {input}

Proposed Bioligical Processes:
{y}

Out of the proposed biological processes, choose the most accurate biological process.
'''


vote_prompt = '''Given a set of genes and proposed biological processes describing the system, your task is to vote on the two best biological process describing the system. Biological processes are organized in a hierarchical ontology, and the most general biological processes are at the top of the hierarchy. For example, "cellular process" is a general biological process, and "cellular response to stimulus" is a more specific biological process. You should vote on the best biological process.

Here is the set of genes:
Genes: {input}

Here are the biological processes for you to vote on:
Biological Processes: {choice}

Given the set of genes and the proposed biological processes, vote on the two best biological processes.
'''


GO_Enrich_prompt = '''
Here are the processes we've discussed: 
{y}

Enrichr has proposed these processes: {terms}. 
Could you use these processes to compare against the proposed biological processes mentioned above, name the differences between them, and provide a summary? 
Conclude, out of all the processes, including processes by Enrichr, which process is the best. 

Please answer strictly in JSON format:
{format_4}
'''

format_4 = {'Analysis': [
            {'Biological Process': 'Your First Biological Process', 
             'Comparison': 'Comparing the biological processes to terms proposed by Enrichr'}, 
            {'Biological Process': 'Your Second Biological Process', 
             'Comparison': '<Comparing the biological processes to terms proposed by Enrichr>'}], 
             'Best Biological Process': {'Biological Process': '<The best biological Process>'}
           }


criticism_prompt = '''
You are an efficient and insightful assistant to a molecular biologist.

You have been working on a project to generate a brief name for the most prominent biological process performed by the set of proteins listed below.

Proteins: {input}

You have generated different candidate biological processes in an iterative way. Initially, you proposed a biological proposed one or more candidate biological processes. Then, for each candidate biological process, you reflected on the candidate biological process and edited it to become more accurate.

In this process, you decided that the following biological process, generated by the sequence below, was a poor candidate for the most prominent biological process performed by the set of proteins listed above. In one or two sentences, provide a criticism of why this biological process is a poor candidate.

Biological Process Generation Sequence: {omit_y_path}

Please answer strictly in JSON format:
{format_5}
'''

format_5 = {'Criticism': 'Your Criticism'}