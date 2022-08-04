import os, sys
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rouge_score import rouge_scorer
import scipy

import sys
sys.path.append("/home/divy/FiD")

import src
from src.data import load_data
from src.evaluation import ems
import src.model

# helper function to remove specific tokens and separate out 
# and extract only the passages and remove the question
def subfinder(mylist, pattern):
    matches = []
    for i in range(len(mylist)):
        if mylist[i] == pattern[0] and mylist[i:i+len(pattern)] == pattern:
            matches.append((i + len(pattern), pattern))
    return matches

def generate_input_output_scores(model, 
                                    dataset, 
                                    tokenizer, 
                                    collator, 
                                    rouge_threshold=0.7,
                                    topk_passages=5):
    
    # define the scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rouge_similarity_answers = []
    topk_passage_means = []
    topk_passage_stdevs = []


    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=1,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )

    model.eval()

    for i, batch in tqdm(enumerate(dataloader)):
        (idx, labels, _, context_ids, context_mask) = batch

        with torch.no_grad():
            
            generated_output, log_prob = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                do_sample=False,
                max_length=150,
                top_p=0.9,
                temperature=1.0,
                output_confidence=True,

            )

            model_forward = model.forward(
                input_ids=context_ids.cuda(),
        
                decoder_input_ids=generated_output.cuda(),
                output_attentions=True,
                output_unnormalized_attentions=True,
            )

        human_readable_generated_output = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        ground_truth_answer = tokenizer.decode(labels[0], skip_special_tokens=True)

        rouge_score = scorer.score(ground_truth_answer, human_readable_generated_output)['rougeL'].fmeasure

        rouge_similarity_answers.append((ground_truth_answer,
                                            human_readable_generated_output,
                                            rouge_score,
                                            i))
        
        cross_attentions = model_forward.cross_attentions
        stacked_forward_attentions = torch.cat(cross_attentions, dim=0)
        msk = torch.reshape(context_mask, (1, context_mask.shape[1]*context_mask.shape[2])).cuda()

        masked_stacked_forward_attentions = stacked_forward_attentions.masked_fill(msk == False, -10000.0)

        avg_attn_matrix = torch.mean(masked_stacked_forward_attentions, dim=(0, 1)).cpu()
        context_ids_reshaped = torch.reshape(context_ids, (1, context_ids.shape[1]*context_ids.shape[2]))
        all_input_tokens = tokenizer.convert_ids_to_tokens(context_ids_reshaped[0])

        start_pattern = ['▁title', ':', '▁context', ':']
        end_pattern = ['</s>']

        start_index_pattern = subfinder(all_input_tokens, start_pattern)
        end_index_pattern = subfinder(all_input_tokens, end_pattern)

        start_indices = [el[0] for el in start_index_pattern]
        end_indices = [el[0] for el in end_index_pattern]

        relevant_ranges = [(start_index, end_index - 1) for start_index, end_index in zip(start_indices, end_indices)]
        
        # list of passage scores
        mean_savgols = []

        for relevant_range in relevant_ranges:
            spliced_attn_matrix = avg_attn_matrix[:, relevant_range[0]:relevant_range[1]]

            maxpool = torch.max(spliced_attn_matrix, axis=0).values.cpu().numpy()
            m = maxpool
           
            try:
                savgol = scipy.signal.savgol_filter(m, 10, 3)

            except ValueError:
                try:
                    savgol = scipy.signal.savgol_filter(m, 5, 3)

                except ValueError:
                    try:
                        savgol = scipy.signal.savgol_filter(m, 3, 2)

                    except ValueError:
                        try:
                            savgol = scipy.signal.savgol_filter(m, 2, 1)

                        except ValueError:
                            try:
                                savgol = scipy.signal.savgol_filter(m, 1, 0)

                            except ValueError:
                                savgol = [-float('inf')]

            mean_savgol = np.mean(savgol)

            mean_savgols.append(mean_savgol)

        mean_savgols_np = np.array(mean_savgols)
        topk_mean_savgols = np.partition(mean_savgols_np, -topk_passages)[-topk_passages:]
        
        mean_topk_mean_savgols = np.mean(topk_mean_savgols)
        stdev_topk_mean_savgols = np.std(topk_mean_savgols)

        topk_passage_means.append(mean_topk_mean_savgols)
        topk_passage_stdevs.append(stdev_topk_mean_savgols)
    

    return (
        rouge_similarity_answers,
        topk_passage_means,
        topk_passage_stdevs
    )

        


            
            