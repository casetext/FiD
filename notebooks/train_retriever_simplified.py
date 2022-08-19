import os, sys
import torch
import transformers
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import torch.nn as nn
import argparse
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn

import sys
sys.path.append("/home/divy/FiD")

import src
from src.data import load_data
from src.util import average_main
from src.evaluation import ems
import src.model

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio)*step/float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
            1.0 + (self.min_ratio - 1) * (step - self.warmup_steps)/float(max(1.0, self.scheduler_steps - self.warmup_steps)),
        )

def train(model, 
        optimizer, 
        scheduler, 
        global_step,
        train_dataset, 
        collator,
        save_every=500):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, 
        sampler=train_sampler, 
        batch_size=1, 
        drop_last=True, 
        num_workers=10, 
        collate_fn=collator,
    )

    loss, curr_loss = 0.0, 0.0
    epoch = 1
    model.train()
    while global_step < 20000:

        epoch += 1
        progress_bar = tqdm(train_dataloader, leave=True, position=0, disable=False)
        for i, batch in tqdm(enumerate(progress_bar)):
            global_step += 1
            (idx, question_ids, question_mask, passage_ids, passage_mask, gold_score) = batch
            _, _, _, train_loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=passage_ids.cuda(),
                passage_mask=passage_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            train_loss.backward()

            if global_step % 1 == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
            
            progress_bar.set_postfix({"train loss": train_loss.item()})
            # train_loss = src.util.average_main(train_loss, opt)
            curr_loss += train_loss.item()

            if i % save_every == 0:
                print(f"saving model at checkpoint {i} ...")
                model_to_save = model.module if hasattr(model, "module") else model
                path = os.path.join(dir_path, "checkpoints", f"checkpoint_{i}")
                os.makedirs(path, exist_ok=True)

                model_to_save.save_pretrained(path)

            if global_step > 20000:
                break
        

if __name__ == "__main__":

    # load data is in casetext ml datasets
    load_data_path = "/home/divy/casetext-ml-datasets/fid_reader_distill_question_data/notebooks/movant_fid_formatted_train.json"
    dir_path = os.path.dirname(os.path.realpath(__file__))   

    train_examples = load_data(
        load_data_path,
        global_rank=0,
        world_size=1
    )
    
    n_passages = 25
    train_dataset = src.data.Dataset(train_examples, n_passages)

    config = src.model.RetrieverConfig(
        indexing_dimension=768,
        apply_question_mask=True,
        apply_passage_mask=True,
        extract_cls=False,
        projection=True,
    )

    model_class = src.model.Retriever
    model = model_class(config, initialize_wBERT=True)

    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=1000, scheduler_steps=30000, min_ratio=0., fixed_lr=False)

    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    collator_function = src.data.RetrieverCollator(
        tokenizer, 
        passage_maxlength=200, 
        question_maxlength=40
    )

    train(
        model,
        optimizer,
        scheduler,
        0,
        train_dataset,
        collator_function,
    )

    







