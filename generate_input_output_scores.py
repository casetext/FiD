import os, sys
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rouge_score import rouge_scorer
import scipy.signal

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
            
            generated_output, output_confidence = model.generate(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                do_sample=True,
                max_length=150,
                top_p=0.9,
                temperature=0.7,
                output_confidence=True,

            )

            model_forward = model.forward(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
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
                                            output_confidence,
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
        np.array(topk_passage_means),
        np.array(topk_passage_stdevs)
    )

if __name__ == "__main__":
    
    # define the load paths
    nq_path =  "/mnt/disks/external_mounted_disk/datasets/NQ/NQ/test.json"
    compose_path = "/mnt/disks/external_mounted_disk/datasets/compose_FiD/compose_fid_qa/dev.json"

    # preprocess and collate data
    eval_examples = load_data(
                            compose_path,
                            global_rank=0,
                            world_size=1,
                            )
    
    n_passages = 25 
    eval_dataset = src.data.Dataset(eval_examples, n_passages)

    # load the model
    model_class = src.model.FiDT5
    model_load_path = "/home/divy/FiD/model_ckpts/test_experiment_large_fid_qa_compose"
    model = model_class.from_pretrained(model_load_path)
    model = model.cuda()

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    collator = src.data.Collator(200, tokenizer, answer_maxlength=-1)

    (
        rouge_similarity_answers,
        topk_passage_means,
        topk_passage_stdevs
    ) = generate_input_output_scores(
        model,
        eval_dataset,
        tokenizer,
        collator
    )

    print("saving out arrays ...")

    np.save("./numpy_drops/topk_passage_means_p09_t07.npy", topk_passage_means)
    np.save("./numpy_drops/topk_passage_stdevs_p09_t07.npy", topk_passage_stdevs)

    with open("./numpy_drops/rouge_similarity_answers_dev_v2", "wb") as fp:
        pickle.dump(rouge_similarity_answers, fp)

            
            