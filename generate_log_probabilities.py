import os, sys
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler
import argparse
import numpy as np 
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import sys
sys.path.append("/home/divy/FiD")

import src
from src.data import load_data
from src.evaluation import ems
import src.model

def generate_log_sum_probabilities(model, dataset, tokenizer, collator):

    exact_match_log_probabilities = []
    incorrect_log_probabilities = []
    exact_match_answers = []

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
            sequences = model.generate(
                        input_ids=context_ids.cuda(),
                        attention_mask=context_mask.cuda(),
                        max_length=100,
                        return_dict_in_generate=True,
                        output_scores=True
                    ).sequences

            
            for k, o in enumerate(sequences):
                ans = tokenizer.decode(o, skip_special_tokens=True)
                gold = dataset.get_example(idx[k])['answers']
                score = ems(ans, gold)
                
            log_probability = model.obtain_log_generated_probability(
                input_ids=context_ids.cuda(),
                attention_mask=context_mask.cuda(),
                max_length=100

            )
            
            if score == True:
                exact_match_log_probabilities.append(log_probability.cpu().numpy())
                exact_match_answers.append(ans)

            else:
                incorrect_log_probabilities.append(log_probability.cpu().numpy())

    return np.array(exact_match_log_probabilities), np.array(incorrect_log_probabilities), exact_match_answers

if __name__ == "__main__":

    # define the load paths
    nq_path =  "/mnt/disks/external_mounted_disk/datasets/NQ/NQ/test.json"
    compose_path = "/mnt/disks/external_mounted_disk/datasets/compose_FiD/compose_fid_qa/test.json"

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
    model_load_path = "/home/divy/FiD/model_ckpts/fid_t5_largq_tqa_compose"
    model = model_class.from_pretrained(model_load_path)
    model = model.cuda()

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
    collator = src.data.Collator(200, tokenizer, answer_maxlength=-1)

    exact_match_log_probabilities, incorrect_log_probabilites, exact_match_answers = generate_log_sum_probabilities(
        model,
        eval_dataset,
        tokenizer,
        collator
    )

    print("saving out arrays ...")
    np.save("./numpy_drops/exact_matches_test.npy", exact_match_log_probabilities)
    np.save("./numpy_drops/incorrects_test.npy", incorrect_log_probabilites)

    with open("./numpy_drops/exact_answers_test", "wb") as fp:
        pickle.dump(exact_match_answers, fp)


    




 
    