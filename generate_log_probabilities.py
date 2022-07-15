import os, sys
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from rouge_score import rouge_scorer

import sys
sys.path.append("/home/divy/FiD")

import src
from src.data import load_data
from src.evaluation import ems
import src.model

def generate_log_sum_probabilities(model, dataset, tokenizer, collator, rouge_threshold=0.5):

    # define the scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    exact_match_log_probabilities = []
    rouge_similarity_log_probabilities = []
    incorrect_log_probabilities = []

    rouge_similarity_answers = []

    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset,
        sampler=sampler,
        batch_size=1,
        drop_last=False,
        num_workers=10,
        collate_fn=collator
    )

    model.eval()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
        
            (idx, _, _, context_ids, context_mask) = batch
            sequences, log_probability = model.generate(
                        input_ids=context_ids.cuda(),
                        attention_mask=context_mask.cuda(),
                        do_sample=True,
                        max_length=150,
                        top_p=0.9,
                        temperature=1.0,
                        output_confidence=True
                    )

            sequences = sequences.cpu()

            for k, o in enumerate(sequences):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']

                score = ems(ans, gold)

            # compute rouge score
            rouge_score = scorer.score(gold[0], ans)['rougeL'].fmeasure

            if rouge_score > rouge_threshold:
                rouge_similarity_log_probabilities.append(log_probability)
                rouge_similarity_answers.append((gold[0], ans, rouge_score, i))

                if score == True:
                    exact_match_log_probabilities.append(log_probability)
            
            else:
                incorrect_log_probabilities.append(log_probability)
            
    return (    
                np.array(incorrect_log_probabilities),
                np.array(exact_match_log_probabilities),  
                np.array(rouge_similarity_log_probabilities), 
                rouge_similarity_answers
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
    
    n_passages = 20
    eval_dataset = src.data.Dataset(eval_examples, n_passages)

    # load the model
    model_class = src.model.FiDT5
    model_load_path = "/home/divy/FiD/model_ckpts/test_experiment_large_fid_qa_compose"
    model = model_class.from_pretrained(model_load_path)
    model = model.cuda()

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    collator = src.data.Collator(200, tokenizer, answer_maxlength=-1)

    (
        incorrect_log_probabilites, 
        exact_match_log_probabilities,
        rouge_similarity_log_probabilities,  
        rouge_similarity_answers
        
    ) = generate_log_sum_probabilities(
        model,
        eval_dataset,
        tokenizer,
        collator,
        rouge_threshold=0.7
    )

    print("saving out arrays ...")

    np.save("./numpy_drops/incorrects_dev_sampling.npy", incorrect_log_probabilites)
    np.save("./numpy_drops/exact_matches_dev_sampling.npy", exact_match_log_probabilities)
    np.save("./numpy_drops/rouge_matches_dev_sampling.npy", rouge_similarity_log_probabilities)
    
    with open("./numpy_drops/rouge_similarity_answers_dev_sampling", "wb") as fp:
        pickle.dump(rouge_similarity_answers, fp)

    


    




 
    
