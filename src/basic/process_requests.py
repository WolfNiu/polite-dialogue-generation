#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from nltk.tokenize import word_tokenize, TweetTokenizer
from nltk import FreqDist
import logging
import re
from gensim.models import KeyedVectors
import string
import pickle
import random
from nltk.tokenize.stanford import StanfordTokenizer
import argparse

from util import load_pickle, load_pickles, dump_pickles, dump_pickle


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Stanford Politeness Corpus")
    parser.add_argument(
        "--wiki_file", type=str, default="dataset_wikipedia.annotated.pkl",
        help="path to WIKI politeness data file")
    parser.add_argument(
        "--se_file", type=str, default="dataset_stack-exchange.annotated.pkl",
        help="path to SE politeness data file")
    parser.add_argument(
        "--tagger_path", type=str, 
        default="/playpen/home/tongn/stanford-postagger-full-2017-06-09/stanford-postagger.jar",
        help="path to the Stanford pos tagger")
    parser.add_argument(
        "--word2vec", type=str, default="/playpen/home/tongn/GoogleNews-vectors-negative300.bin",
        help="path to pretrained word2vec binary file")
    parser.add_argument(
        "--use_existing_vocab", action="store_true", 
        help="whether to use an existing vocab set")
    args = parser.parse_args()
    return args


"""
Load file
"""
path = data_path = "data/Stanford_politeness_corpus/"
args = parse_args()
filenames = [args.wiki_file, args.se_file]
tagger_path = args.tagger_path
word2vec = args.word2vec
use_existing_vocab = args.use_existing_vocab
files = [
    os.path.join(data_path, filename)
    for filename in filenames]

datasets = load_pickles(files)

"""
Manually correcting a very long sequence in the example
"""
bad = ("There are 78865786736479050355236321393218506229513"
       "597768717326329474253324435944996340334292030428401"
       "198462390417721213891963883025764279024263710506192"
       "662495282993111346285727076331723739698894392244562"
       "145166424025403329186413122742829485327752424240757"
       "390324032125740557956866022603190417032406235170085"
       "879617892222278962370389737472000000000000000000000"
       "0000000000000000000000000000 permutations of 200-el"
       "ement set.  You want them all?")
bad_index = datasets[1].index(bad)
datasets[1][bad_index] = (
    "There are UNK_TOKEN permutations of "
    "200-element set. You want them all?")

print("Tokenizing all requests.")

tweet_tokenizer = TweetTokenizer(
    preserve_case=True, reduce_len=True, strip_handles=True)

tokenized_datasets_original_tweet = [
    [tweet_tokenizer.tokenize(request)
     for request in dataset]
    for dataset in datasets]

print("Retokenizing with Stanford tokenizer. This may take a long time.")

path_pos = "/playpen/home/tongn/stanford-postagger-full-2017-06-09/"
jar_pos = "stanford-postagger.jar"

tokenizer = StanfordTokenizer(path_pos + jar_pos)
tokenizer = StanfordTokenizer(tagger_path)

tokenized_datasets_original = [
    [tokenizer.tokenize(' '.join(request).strip())
     for request in dataset]
    for dataset in tokenized_datasets_original_tweet]
# tokenized_datasets_original = tokenized_datasets_original_tweet

"""
Convert all tokens to lowercase
"""
tokenized_datasets = [
    [[token.lower()
      for token in request]
     for request in dataset]
    for dataset in tokenized_datasets_original]

"""
Build the whole vocabulary

Vocab lists:
• special token: "UNK_TOKEN"
• vocab_shared: intersection of word2vec vocab and politeness vocab
• vocab_freq: frequent vocab that is not in word2vec vocab
"""

UNK = "UNK_TOKEN"

if use_existing_vocab:
    vocab_politeness = load_pickle("data/Stanford_politeness_corpus/vocab_politeness.pkl")
else:
    # Load word embedding model
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = KeyedVectors.load_word2vec_format(fname=word2vec, binary=True)
    
    freq_threshold = 2

    all_tokens = [token
        for dataset in tokenized_datasets
        for request in dataset
        for token in request]

    fdist = FreqDist(all_tokens)
    fdist_lst = fdist.most_common()
    vocab_politeness = [token for (token, _) in fdist_lst]
    vocab_politeness_freq = [
        token 
        for (token, freq) in fdist_lst 
        if (freq >= freq_threshold)]

    vocab_word2vec = list(model.vocab) # get word2vec vocabulary list
    vocab_shared = list((set(vocab_politeness)).intersection(set(vocab_word2vec)))
    vocab_new = list((set(vocab_politeness_freq)).difference(set(vocab_word2vec)))
    vocab_politeness = [UNK] + vocab_new + vocab_shared

    print("Shared vocab size: %d" % len(vocab_shared))
    print("New vocab size: %d" % len(vocab_new))

    """
    Obtain the reduced word2vec embedding matrix
    """
    embedding_word2vec = model[vocab_shared]

"""
Create dictionaries between indices and tokens
"""
index2token = {i: token for (i, token) in enumerate(vocab_politeness)}
token2index = {token: i for (i, token) in enumerate(vocab_politeness)}

"""
Replace a token with its index in the vocab
"""
index_UNK = token2index[UNK]

def replace_with_index(token):
#     if token in vocab_politeness:
    try:
        return token2index[token]
#     else:
    except:
        return index_UNK


print("Start indexing dataset... This may take a while")
indexed_datasets = [
    [[replace_with_index(token)
      for token in request]
     for request in dataset]
    for dataset in tokenized_datasets]

if use_existing_vocab:
    lsts = [indexed_datasets[0], indexed_datasets[1]]
    pickle_lst = ["dataset_WIKI", "dataset_SE"]
else:
    """
    Pickle all lists
    """
    lsts = [
        vocab_politeness,
        vocab_shared,
        vocab_new,
        indexed_datasets[0], indexed_datasets[1],
        embedding_word2vec]

    pickle_lst = [
        "vocab_politeness",
        "shared_vocab_politeness",
        "new_vocab_politeness",
        "dataset_WIKI", "dataset_SE",
        "embedding_word2vec_politeness"]

pickle_files = [
    os.path.join(data_path, file + ".pkl")
    for file in pickle_lst]

dump_pickles(pickle_files, lsts)

