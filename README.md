# BERT for Evidence Retrieval and Claim Verification

## This is a repository for the code we used in "BERT for Evidence Retrieval and Claim Verification". The paper has been accepted in ECIR and would be published soon. Meanwhile, you can see the older version here: https://arxiv.org/abs/1910.02655

This repository is still updating. Meanwhile, I've added the hyperparameters.

The original Fever dataset repository: https://github.com/sheffieldnlp/fever-naacl-2018
UKP-Athene repository that we used for document retrieval: https://github.com/UKPLab/fever-2018-team-athene

### Requirements
* Python 3.5.6
* Pytorch 1.1.0
* Pytorch-pretrained-bert  0.6.2 (HuggingFace BERT: https://github.com/huggingface/transformers)
* NLTK 3.4.1

### Initial Steps
1. Get the dataset from the [Fever dataset repository] (https://github.com/sheffieldnlp/fever-naacl-2018)
1. Get the dataset ready using the [UKP-Athene codes] (https://github.com/UKPLab/fever-2018-team-athene)

### Evidence Retrieval

    python -u run_classifier_sentence_retrieval.py \
    --task_name Fever \
    --do_train \
    --do_eval \
    --data_dir ./ \
    --do_lower_case \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 16 \
    --negative_batch_size 64 \
    --losstype cross_entropy_mining \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir /dir
    

    
    python -u run_classifier_sentence_retrieval_pairs.py \
    --task_name Fever \
    --do_train \
    --do_eval \
    --data_dir ./ \
    --do_lower_case \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 16 \
    --losstype ranknet \
    --learning_rate 2e-5 \
    --num_train_epochs 1.0 \
    --output_dir /dir


### Claim Verification

    python -u run_classifier_negativemining.py \
    --task_name Fever \
    --do_train \
    --do_eval \
    --data_dir ./ \
    --do_lower_case \
    --bert_model bert-base-uncased \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --negative_batch_size 128 \
    --learning_rate 2e-5 \
    --num_train_epochs 2.0 \
    --output_dir

### Contacts
If you have any questions or problems regarding the code, please don't hesitate to email me {a.soleimani}@uva.nl
