import time
import sys
import torch
import transformers
from pathlib import Path
import numpy as np
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
from tqdm import tqdm

import src.slurm
import src.util
import src.evaluation
import src.data
import src.model
from src.options import Options

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

    # dir_path = Path(opt.checkpoint_dir)/opt.name
    # directory_exists = dir_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    # dir_path.mkdir(parents=True, exist_ok=True)
    # if not directory_exists and opt.is_main:
    #     options.print_options(opt)

    #Load data
    # try to load legal bert model
    load_path = '/home/divy/FiD/prod_ranker_20220409'
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-cased')

    # tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
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
    config = src.model.RetrieverConfig(
        indexing_dimension=opt.indexing_dimension,
        apply_question_mask=not opt.no_question_mask,
        apply_passage_mask=not opt.no_passage_mask,
        extract_cls=opt.extract_cls,
        projection=not opt.no_projection,
    )
    model_class = src.model.Retriever
    model = model_class.from_pretrained("/home/divy/FiD/checkpoint_fid_bertbase/experiment_name/checkpoint/best_dev")

    # model = model_class(config, initialize_wBERT=True)

    model.proj = torch.nn.Linear(768, 256)
    model.norm = torch.nn.LayerNorm(256)
    model.config.indexing_dimension = 256
    model = model.to(opt.device)

    loss, inversions, avg_topk, idx_topk = evaluate(
        model,
        eval_dataset,
        collator_function,
        opt
    )

    print(loss)
    print(inversions)
    print(avg_topk)
    print(idx_topk)