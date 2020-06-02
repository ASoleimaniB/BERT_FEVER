# BERT for Evidence Retrieval and Claim Verification

## This is a repository for the code we used in "BERT for Evidence Retrieval and Claim Verification". The paper has been published in ECIR2020: https://link.springer.com/chapter/10.1007/978-3-030-45442-5_45


The original Fever dataset repository: https://github.com/sheffieldnlp/fever-naacl-2018
UKP-Athene repository that we used for document retrieval: https://github.com/UKPLab/fever-2018-team-athene

### Requirements
* Python 3.5.6
* Pytorch 1.1.0
* Pytorch-pretrained-bert  0.6.2 (HuggingFace BERT: https://github.com/huggingface/transformers)
   * pip install pytorch-pretrained-bert==0.6.2
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


### Data
You need the following datasets to train and perform validation but due to copyright issues we cannot upload the dataset. 
You should get it from the Fever website and then use Athene's code or any other document retrieval approach to get all the potential sentences in the retrieved documents.
* train_sentences_pos.tsv : has all the positive pairs 
* train_sentences_neg_32.tsv : has a number of negative samples 
* dev_sentences.tsv : development set

### Claim Verification

    python -u run_classifier_negativemining.py (or run_classifier_ret)\
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
