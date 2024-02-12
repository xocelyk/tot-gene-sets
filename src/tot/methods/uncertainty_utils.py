import json
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
from collections import Counter, defaultdict
    
import numpy as np
import json

SapBERT_tokenizer = AutoTokenizer.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
SapBERT_model = AutoModel.from_pretrained('cambridgeltl/SapBERT-from-PubMedBERT-fulltext')

def group_and_frequency_analyze_by_similarity(ys_list, top_n=3, exploration_rate=0.1):
    # Flatten the list to process all items, including duplicates
    ys = [item for items in ys_list for item in items]
    all_items_flat = [json.loads(item)['Biological Process'] for items in ys_list for item in items]

    # Get embeddings for all items, considering duplicates
    all_items_embeddings = getSentenceEmbedding(all_items_flat, SapBERT_tokenizer, SapBERT_model)
    # Calculate cosine similarity between all items
    similarity_matrix = cosine_similarity(all_items_embeddings)

    # Group items based on similarity, considering threshold
    threshold = 0.98
    groups = []
    for i in range(len(all_items_flat)):
        grouped = False
        for group in groups:
            if any(similarity_matrix[i][j] > threshold for j in group):
                group.append(i)
                grouped = True
                break
        if not grouped:
            groups.append([i])

    item_counts = sum([len(group) for group in groups])
    # Calculate group frequencies
    group_frequencies = {i: len(group) / item_counts for i, group in enumerate(groups)}

    # Adjust frequencies with exploration factor and store adjustments
    adjustments = {i: np.random.rand() * exploration_rate for i in group_frequencies.keys()}
    adjusted_group_frequencies = {i: freq + adjustments[i] for i, freq in group_frequencies.items()}
    print(f'group_frequencies: {group_frequencies}')
    print(f'adjusted_group_frequencies: {adjusted_group_frequencies}')
    # Sort groups by adjusted frequency
    sorted_group_freq = sorted(adjusted_group_frequencies.items(), key=lambda x: x[1], reverse=True)

    # Select top groups based on adjusted frequencies
    top_groups = sorted_group_freq[:top_n]

    new_ys = []
    frequencies = []
    adjusted_frequencies = []
    for index, _ in top_groups:
        ys_index = groups[index][0]
        new_ys.append(ys[ys_index])
        frequencies.append(group_frequencies[index])  # Original frequency
        adjusted_frequencies.append(adjusted_group_frequencies[index])  # Adjusted frequency
        print(f"Group {index + 1}: {', '.join(all_items_flat[i] for i in groups[index])}")

    grouped_items_by_index = []
    for index, _ in sorted_group_freq:
        grouped_items_by_index.append([
            {"index": index},
            {"grouped_names": ', '.join(all_items_flat[i] for i in groups[index])},
            {"original_frequency": group_frequencies[index]},
            {"adjusted_frequency": adjusted_group_frequencies[index]}
        ])

    return new_ys, frequencies, adjusted_frequencies, grouped_items_by_index




def getSentenceEmbedding(sentence, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        
    # Perform pooling. In this case, mean pooling.
    sentence_embedding = model_output.last_hidden_state.mean(dim=1)#mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embedding


#depreciated
    
# def group_and_frequency_analyze_by_similarity(ys_list, top_n=3, e):
    
#     # Flatten the list to process all items, including duplicates 
#     ys = [item for items in ys_list for item in items]
#     all_items_flat = [json.loads(item)['Biological Process'] for items in ys_list for item in items]


#     # Get embeddings for all items, considering duplicates 
#     all_items_embeddings = getSentenceEmbedding(all_items_flat, SapBERT_tokenizer, SapBERT_model)
#     # Calculate cosine similarity between all items
#     similarity_matrix = cosine_similarity(all_items_embeddings)


#     # Group items based on similarity, considering threshold
#     threshold = 0.90
#     groups = []
#     for i in range(len(all_items_flat)):
#         grouped = False
#         for group in groups:
#             if any(similarity_matrix[i][j] > threshold for j in group):
#                 group.append(i)
#                 grouped = True
#                 break
#         if not grouped:
#             groups.append([i])

#     item_counts = sum([len(group) for group in groups])
#     # Calculate group frequencies
#     group_frequencies = {i: len(group)/item_counts for i, group in enumerate(groups)}

#     # Sort groups by frequency
#     sorted_group_freq = sorted(group_frequencies.items(), key=lambda x: x[1], reverse=True)

#     # Select top 3 groups
#     top_3_groups = sorted_group_freq[:top_n]

#     new_ys = []
#     frequencies = []
#     # Display top 3 grouped names and frequencies
# #     print("1) Top three grouped names based on frequency:")
# #     print("\n2) Top three frequencies:")
#     for index, freq in top_3_groups:
#         ys_index = groups[index][0]
#         new_ys.append(ys[ys_index])
#         frequencies.append(freq)
#         print(f"Group {index + 1}: {', '.join(all_items_flat[i] for i in groups[index])}")
# #         print(freq)        

#     grouped_items_by_index = []
#     # Display all grouped names and their frequencies
# #     print("\n3) All grouped names and their frequency:")
#     for index, freq in sorted_group_freq:
#         grouped_items_by_index.append([{index}, {', '.join(all_items_flat[i] for i in groups[index])}, {freq}])


#     return new_ys, frequencies, grouped_items_by_index