import time
import os, sys
import torch
import transformers
from pathlib import Path
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm
import json

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options
import ipdb

# DRM: THIS SCRIPT HAS NOT BEEN TESTED ON MULTIPLE GPUS, PROCEED WITH CAUTION IF YOU TAKE THAT ROUTE.

def evaluate(model, dataset, collator, opt):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset, 
        sampler=sampler, 
        batch_size=opt.per_gpu_batch_size,
        drop_last=False, 
        num_workers=10, 
        collate_fn=collator
    )
    model.eval()
    if hasattr(model, "module"):
        model = model.module
    total = 0
    eval_loss = []

    avg_topk = {k:[] for k in [1, 2, 5] if k <= opt.n_context}
    idx_topk = {k:[] for k in [1, 2, 5] if k <= opt.n_context}
    inversions = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, leave=True, position=0, disable=False)
        for i, batch in enumerate(progress_bar):
            (idx, question_ids, question_mask, context_ids, context_mask, gold_score) = batch

            _, _, scores, loss = model(
                question_ids=question_ids.cuda(),
                question_mask=question_mask.cuda(),
                passage_ids=context_ids.cuda(),
                passage_mask=context_mask.cuda(),
                gold_score=gold_score.cuda(),
            )

            src.evaluation.eval_batch(scores, inversions, avg_topk, idx_topk)
            total += question_ids.size(0)

    inversions = src.util.weighted_average(np.mean(inversions), total, opt)[0]
    for k in avg_topk:
        avg_topk[k] = src.util.weighted_average(np.mean(avg_topk[k]), total, opt)[0]
        idx_topk[k] = src.util.weighted_average(np.mean(idx_topk[k]), total, opt)[0]

    return loss, inversions, avg_topk, idx_topk
    
if __name__ == "__main__":

    options = Options()
    options.add_retriever_options()
    options.add_optim_options()
    opt = options.parse()
    torch.manual_seed(opt.seed)
    src.slurm.init_distributed_mode(opt)
    src.slurm.init_signal_handler()


    if opt.is_distributed:
        torch.distributed.barrier()

    # toggle between evaluating legal bert and bert-base-uncased
    evaluate_legal_bert = True

    #Load data
    # try to load legal bert model

    if evaluate_legal_bert:
        load_path = os.path.join(os.getcwd(),'prod_ranker_20220409')
        tokenizer = transformers.BertTokenizerFast.from_pretrained(load_path)
    
    else:
        tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    collator_function = src.data.RetrieverCollator(
        tokenizer, 
        passage_maxlength=opt.passage_maxlength, 
        question_maxlength=opt.question_maxlength
    )

    eval_examples = src.data.load_data(
        opt.eval_data, 
        global_rank=opt.global_rank, 
        world_size=opt.world_size, 
    )
    eval_dataset = src.data.Dataset(eval_examples, opt.n_context)

    global_step = 0
    best_eval_loss = np.inf
    
    # edit the model path here, accordingly

    if evaluate_legal_bert:
        model_path = os.path.join(os.getcwd(), "checkpoint_lawbert/experiment_name/checkpoint/step-600")
    
    else:
        model_path = os.path.join(os.getcwd(), "checkpoint_bert_base/experiment_name/checkpoint/step-700")

    # load model checkpoint
    model_class = src.model.Retriever

    model = model_class.from_pretrained(model_path)

    model = model.to(opt.device)

    loss, inversions, avg_topk, idx_topk = evaluate(
        model,
        eval_dataset,
        collator_function,
        opt
    )

    print(f"eval loss: {loss}")
    print(f"inversions: {inversions}")
    print(f"avg_topk: {avg_topk}")
    print(f"idx_topk: {idx_topk}")

   