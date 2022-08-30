This repository contains code for:
- Fusion-in-Decoder models
- Distilling Knowledge from Reader to Retriever

## Additional work, with legal data

***Note: we did not test the below code in a multi-gpu setting, proceed with caution if you wish to go that route.***

We report results of the performance of the retriever using the paper's metrics (number of `inversions`, `avg_topk` and `idx_topk`). 

To run the code, the following steps must be performed to prep it:

- obtain the production ranker from `gs://siamese-models/prod_ranker_20220409`.
- in `src/model.py`, modify the `Retriever` class to load the default model (under `if initialize_wBERT`) from (replacing `/path/to` with your path)
```
self.model = transformers.BertModel.from_pretrained('/path/to/prod_ranker_20220409')
```
- obtain the train/test data from [here](https://github.com/casetext/casetext-ml-datasets/tree/main/fid_reader_distill_question_data/notebooks)

- Train a retriever model via (replacing `/path/to` with your path)
```
python train_retriever.py --lr 1e-4 --optim adamw --scheduler linear --train_data /path/to/compose_fid_train_scored.json --eval_data /path/to/compose_fid_dev_scored.json --n_context 25 --total_steps 20000 --scheduler_steps 3000 --checkpoint_dir './checkpoint_lawbert' --eval_freq 100 --save_freq 100
```
- Finally, to evaluate, run 
```
python evaluate_retriever.py --eval_data /home/divy/casetext-ml-datasets/fid_reader_distill_question_data/notebooks/compose_fid_dev_scored.json --n_context 25
```
after modifying line 111 of `evaluate_retriever.py` with the model checkpoint you wish to evaluate.

**Results**

On this data, we found that evaluation performance began to degrade after about 600 steps (equivalent to 700 examples, since `batch_size = 1` by default). Here are the results:

Retriever at zero steps:
```
eval loss: 0.05155069753527641
inversions: 62.80829015544042
avg_topk: {1: 0.031088082901554404, 2: 0.06476683937823834, 5: 0.25129533678756477}
idx_topk: {1: 10.922279792746114, 2: 12.10880829015544, 5: 13.318652849740932}
```

retriever at 600 steps:
```
eval loss: 0.006813043262809515
inversions: 23.564766839378237
avg_topk: {1: 0.5595854922279793, 2: 0.5919689119170984, 5: 0.6580310880829016}
idx_topk: {1: 2.6088082901554404, 2: 4.536269430051814, 5: 8.704663212435234}
```

While this knowledge distillation methodology is internally consistent and improves according to the paper's evaluation metrics, we found that (a) this method does not generalise well to other datasets (such as `cognia`, or even the non-QA `compose`), and in fact performance worsens based on metrics such as MRR/NDCG.

Fundamentally, this training method uses FiD's input scores as ground truths of what relevant vs irrelevant passages are -- via what FiD pays most attention to. And the paper's evaluation metrics measure just that. However, what FiD pays most attention to might not necessarily to what's ***actually*** most relevant, which is why we suspect that this method does not generalise.

## Dependencies

- Python 3
- [PyTorch](http://pytorch.org/) (currently tested on version 1.6.0)
- [Transformers](http://huggingface.co/transformers/) (**version 3.0.2**, unlikely to work with a different version)
- [NumPy](http://www.numpy.org/)


# Data

### Download data
NaturalQuestions and TriviaQA data can be downloaded using [`get-data.sh`](get-data.sh). Both datasets are obtained from the original source and the wikipedia dump is downloaded from the [DPR](https://github.com/facebookresearch/DPR) repository. In addition to the question and answers, this script retrieves the Wikipedia passages used to trained the released pretrained models.

### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id, optional
- `question`: question text
- `target`: answer used for model training, if not given, the target is randomly sampled from the 'answers' list
- `answers`: list of answer text for evaluation, also used for training if target is not given
- `ctxs`: a list of passages where each item is a dictionary containing
        - `title`: article title
        - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'What element did Marie Curie name after her native land?',
  'target': 'Polonium',
  'answers': ['Polonium', 'Po (chemical element)', 'Po'],
  'ctxs': [
            {
                "title": "Marie Curie",
                "text": "them on visits to Poland. She named the first chemical element that she discovered in 1898 \"polonium\", after her native country. Marie Curie died in 1934, aged 66, at a sanatorium in Sancellemoz (Haute-Savoie), France, of aplastic anemia from exposure to radiation in the course of her scientific research and in the course of her radiological work at field hospitals during World War I. Maria Sk\u0142odowska was born in Warsaw, in Congress Poland in the Russian Empire, on 7 November 1867, the fifth and youngest child of well-known teachers Bronis\u0142awa, \"n\u00e9e\" Boguska, and W\u0142adys\u0142aw Sk\u0142odowski. The elder siblings of Maria"
            },
            {
                "title": "Marie Curie",
                "text": "was present in such minute quantities that they would eventually have to process tons of the ore. In July 1898, Curie and her husband published a joint paper announcing the existence of an element which they named \"polonium\", in honour of her native Poland, which would for another twenty years remain partitioned among three empires (Russian, Austrian, and Prussian). On 26 December 1898, the Curies announced the existence of a second element, which they named \"radium\", from the Latin word for \"ray\". In the course of their research, they also coined the word \"radioactivity\". To prove their discoveries beyond any"
            }
          ]
}
```

# Pretrained models.

Pretrained models can be downloaded using [`get-model.sh`](get-model.sh). Currently availble models are [nq_reader_base, nq_reader_large, nq_retriever, tqa_reader_base, tqa_reader_large, tqa_retriever].

```shell
bash get-model.sh -m model_name
```

Performance of the pretrained models:

<table>
  <tr><td>Mode size</td><td colspan="2">NaturalQuestions</td><td colspan="2">TriviaQA</td></tr>
  <tr><td></td><td>dev</td><td>test</td><td>dev</td><td>test</td></tr>
  <tr><td>base</td><td>49.2</td><td>50.1</td><td>68.7</td><td>69.3</td></tr>
  <tr><td>large</td><td>52.7</td><td>54.4</td><td>72.5</td><td>72.5</td></tr>
</table>



# I. Fusion-in-Decoder

Fusion-in-Decoder models can be trained using [`train_reader.py`](train_reader.py) and evaluated with [`test_reader.py`](test_reader.py).

### Train

[`train_reader.py`](train_reader.py) provides the code to train a model. An example usage of the script is given below:

```shell
python train_reader.py \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
```

Training these models with 100 passages is memory intensive. To alleviate this issue we use checkpointing with the `--use_checkpoint` option. Tensors of variable sizes lead to memory overhead. Encoder input tensors have a fixed size by default, but not the decoder input tensors. The tensor size on the decoder side can be fixed using `--answer_maxlength`. The large readers have been trained on 64 GPUs with the following hyperparameters:

```shell
python train_reader.py \
        --use_checkpoint \
        --lr 0.00005 \
        --optim adamw \
        --scheduler linear \
        --weight_decay 0.01 \
        --text_maxlength 250 \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --total_step 15000 \
        --warmup_step 1000 \
```

### Test

You can evaluate your model or a pretrained model with [`test_reader.py`](test_reader.py). An example usage of the script is provided below.

```shell
python test_reader.py \
        --model_path checkpoint_dir/my_experiment/my_model_dir/checkpoint/best_dev \
        --eval_data eval_data.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```



# II. Distilling knowledge from reader to retriever for question answering
This repository also contains code to train a retriever model following the method proposed in our paper: Distilling knowledge from reader to retriever for question answering. This code is heavily inspired by the [DPR codebase](https://github.com/facebookresearch/DPR) and reuses parts of it. The proposed method consists in several steps:

### 1. Obtain reader cross-attention scores
Assuming that we have already retrieved relevant passages for each question, the first step consists in generating cross-attention scores. This can be done using the option `--write_crossattention_scores` in [`test.py`](test.py). It saves the dataset with cross-attention scores in `checkpoint_dir/name/dataset_wscores.json`. To retrieve the initial set of passages for each question, different options can be considered, such as DPR or BM25.

```shell
python test.py \
        --model_path my_model_path \
        --eval_data data.json \
        --per_gpu_batch_size 4 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
        --write_crossattention_scores \
```

### 2. Retriever training

[`train_retriever.py`](train_retriever.py) provides the code to train a retriever using the scores previously generated.

```shell
python train_retriever.py \
        --lr 1e-4 \
        --optim adamw \
        --scheduler linear \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --n_context 100 \
        --total_steps 20000 \
        --scheduler_steps 30000 \
```


### 3. Knowldege source indexing

Then the trained retriever is used to index a knowldege source, Wikipedia in our case.

```shell
python3 generate_retriever_embedding.py \
        --model_path <model_dir> \ #directory
        --passages passages.tsv \ #.tsv file
        --output_path wikipedia_embeddings \
        --shard_id 0 \
        --num_shards 1 \
        --per_gpu_batch_size 500 \
```

### 4. Passage retrieval

After indexing, given an input query, passages can be efficiently retrieved:


```shell
python passage_retrieval.py \
    --model_path <model_dir> \
    --passages psgs_w100.tsv \
    --data_path data.json \
    --passages_embeddings "wikipedia_embeddings/wiki_*" \
    --output_path retrieved_data.json \
    --n-docs 100 \
```

We found that iterating the four steps here can improve performances, depending on the initial set of documents.


## References

[1] G. Izacard, E. Grave [*Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering*](https://arxiv.org/abs/2007.01282)

```bibtex
@misc{izacard2020leveraging,
      title={Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering},
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2007.01282},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

[2] G. Izacard, E. Grave [*Distilling Knowledge from Reader to Retriever for Question Answering*](https://arxiv.org/abs/2012.04584)

```bibtex
@misc{izacard2020distilling,
      title={Distilling Knowledge from Reader to Retriever for Question Answering},
      author={Gautier Izacard and Edouard Grave},
      year={2020},
      eprint={2012.04584},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

See the [LICENSE](LICENSE) file for more details.
