# Reinforcement Learning for Non-Autoregressive Neural Machine Translation

## Clone extension repo
```sh
!git clone https://github.com/jjhlgraber/fairseq_easy_extend.git
cd fairseq_easy_extend
```

## Installing env
```sh
conda create -n NMT python=3.8
conda activate NMT
pip install -r requirements.txt -r "# Torch packages"
pip install -r requirements.txt -r "# fairseq"
```

## Finetuning example for bleu
```sh
python train.py \
   --config-dir "fairseq_easy_extend/models/nat" \
   --config-name "cmlm_config.yaml" \
   criterion.sentence_level_metric=BLEU4 \
   task.data=/home/job/Documents/NLP2/NA-NMT-Project/iwslt14.tokenized.de-en \
   checkpoint.restore_file=/home/job/Documents/NLP2/NA-NMT-Project/checkpoint_best.pt \
   checkpoint.reset_optimizer=True
```

## Requirements

This repository is an example of [fairseq](https://github.com/facebookresearch/fairseq) extension for NLP2 course. 
Please follow the installation instruction of `fairseq`

0. Get access to project drive (ask your TA)
1. Read project description pdf NLP2_ET_2023.pdf and suggested papers
2. Follow GroupC-fairseq notebook in order to get familiar with running model training and evaluation
   1. data for training is in iwslt14 folder
   2. you have pretrained checkpoint at checkpoint_best.pt
   3. if you are not familiar with NMT you can read https://evgeniia.tokarch.uk/blog/neural-machine-translation/
   4. Some notes on fairseq extension https://evgeniia.tokarch.uk/blog/extending-fairseq-incomplete-guide/
3. Objective implementation:
   1. check `rl_criterion.py` in `criterion` folder it gives you a hint how to start working on your objective
   2. Nice explanation of RL for NMT https://www.cl.uni-heidelberg.de/statnlpgroup/blog/rl4nmt/
   3. You can pick any metric and import any library of your choice
4. Run the training with your new objective function:
   1. You can strat fine-tuning to get better/faster results, find `checkpoint_best.pt` in the drive
   2. set `criterion._name` to the name of your implemented criterion
   3. It's enough to fine-tune for <1k steps

