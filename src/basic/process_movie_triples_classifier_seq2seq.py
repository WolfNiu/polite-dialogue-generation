
# coding: utf-8

# In[4]:


"""
Add in a list,
it has indicies for each example in the test set (test set is always ordered)
if it is X-Y, then [X] is stored as list element
if it is X-Y-X, then [X, Y] is stored as list element
"""

import numpy as np
import csv
import pickle
import os
import logging
from nltk import FreqDist
from gensim.models import KeyedVectors
from nltk.tokenize import TweetTokenizer
import re
import argparse

from util import (unzip_lst, group_lst, 
                  decode2string, 
                  load_pickle, load_pickles, 
                  dump_pickle, dump_pickles, 
                  write_lines, read_lines,
                  prepend, append)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess MovieTriples dataset")
    parser.add_argument(
        "--data_path", type=str, default="../data/MovieTriples/",
        help="path to the MovieTriples dataset")
    parser.add_argument(
        "--word2vec", type=str, default="../GoogleNews-vectors-negative300.bin",
        help="path to pretrained word2vec binary file")
    args = parser.parse_args()
    return args



args = parse_args()
data_path = args.data_path
word2vec = args.word2vec



"""
Change the tokenization of MovieTriples so that
it conforms to Stanford Tokenizer
"""

retokenize_tuples = [
    ("<person> ' t", "do n't"),
    ("didn't", "did n't"),
    ("' s", "'s"), # He ' s --> He 's
    ("' re", "'re"), # You ' re --> You 're
    ("' m", "'m"),
    ("' d", "'d"),
    ("' ve", "'ve"),
    ("' ll", "'ll"), 
    ("''", '"'),
    ("``", '"'),
    ("o ' clock", "o'clock"),
    (" dont ", "do n't"),
    (" cant ", "ca n't"),
    ("question:", "question :"),
    ("thankyou", "thank you"),
    ("s . a . c", "s.a.c"), # end of current pre-processing
    ("' em", "'em"), # short for "them"
    ("ma ' am", "ma'am"),
    ("whatll ", "what 'll"),
    ("wan na", "wanna"),
    ("tryi n'to", "tryin' to"),
    ("i t 's", "it 's"),
    (" he most wonderful girl", " the most wonderful girl"),
    ("c . i . a .", "c.i.a."),
    ("k . g . b .", "k.g.b."),
    ("ido n'tcare", "i do n't care"),
    ("`", "'"),
    ("sorewhere", "somewhere")
]
retokenize_reg_tuples = [
    (r"([a-z]+)n ' t", r"\1 n't"),
    (r'"([a-z]+)', r'" \1'),
    (r'([a-z]+)"', r'\1 "'),
    (r"([a-z]+)in ' ", r"\1in' "),
    (r"([a-z]+)in '([a-z]+)", r"\1in' \2")
]

def retokenize(string, tuples=[], reg_tuples=[]):
    retokenized_string = string
    for tup in retokenize_tuples:
        retokenized_string = retokenized_string.replace(*tup)
    for tup in retokenize_reg_tuples:
        retokenized_string = re.sub(*tup, retokenized_string)
    return retokenized_string


"""
Load tab separated datasets
"""

target_path = "../data/MovieTriples/"

filenames = [
    "Training_Shuffled_Dataset.txt", 
    "Validation_Shuffled_Dataset.txt", 
    "Test_Shuffled_Dataset.txt"]

datasets = []
for filename in filenames:
    file = os.path.join(data_path, filename)
    with open(file, "r") as tsv:
        triples_lst = []
        for line in csv.reader(tsv, delimiter="\t"):
            tokenized_line = [
                retokenize(turn, retokenize_tuples, retokenize_reg_tuples).strip().split() # correct a bug in MovieTriples
                for turn in line]
            triples_lst.append(tokenized_line)
    print("Done loading file %s" % file)
    datasets.append(triples_lst)

politeness_path = "../data/politeness"
politeness_filenames = [
    "vocab_politeness.pkl", "shared_vocab_politeness.pkl",
    "new_vocab_politeness.pkl"]
politeness_files = [
    politeness_path + filename
    for filename in politeness_filenames]

[vocab_politeness, shared_vocab_politeness, 
 new_vocab_politeness] = load_pickles(politeness_files)


def get_thresholds(lst, percentile):
    lower = np.percentile(lst, percentile).astype(np.int32)
    upper = np.percentile(lst, 100 - percentile).astype(np.int32)
    return (lower, upper)


percentile = 20

target_lengths = [
    len(turn) 
    for dataset in datasets
    for line in dataset
    for turn in line[1:]]

(lower_threshold_target, upper_threshold_target) = get_thresholds(target_lengths, percentile)

print("Lower target length threshold: %d" % lower_threshold_target)
print("Upper target length threshold: %d" % upper_threshold_target)


# force:
(lower_threshold_target, upper_threshold_target) = (5, 32)


# In[10]:


end_token = "END_TOKEN"


# In[11]:


truncated_datasets = []
source_test_text = []

for (i, dataset) in enumerate(datasets):
    source_lst = []
    target_lst = []
    
    for line in dataset:
        if len(line) in [2, 3]:            
            if (len(line[1]) <= upper_threshold_target 
                and len(line[1]) >= lower_threshold_target):
                source_lst.append(line[0] + [end_token])
                target_lst.append(line[1] + [end_token])
                if i == 2: # if it is the test set
                    source_test_text.append("X: " + ' '.join(line[0]))
                    source_test_text.append("")
        if (len(line) == 3
            and len(line[2]) <= upper_threshold_target 
            and len(line[2]) >= lower_threshold_target):
            source_lst.append(line[0] + [end_token] + line[1] + [end_token]) # concatenate the first two turns
            target_lst.append(line[2] + [end_token])
            if i == 2:
                source_test_text.append("X: " + ' '.join(line[0]))
                source_test_text.append("Y: " + ' '.join(line[1]))

    truncated_datasets.append([source_lst, target_lst])


# In[12]:


print(len(datasets[0]), len(truncated_datasets[0][0]))


# In[13]:


source_test_text_file = target_path + "source_test_text.txt"
write_lines(source_test_text_file, source_test_text)


# In[ ]:


# Load word embedding model
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# model_path = './GoogleNews-vectors-negative300.bin'
# model = KeyedVectors.load_word2vec_format(fname=model_path, binary=True)
model = KeyedVectors.load_word2vec_format(fname=word2vec, binary=True)


# In[ ]:


"""
To extend classifier vocab on a new corpus, say MovieTriples:
(0) get vocab_movie_shared, get vocab_politeness
(1) get vocab_movie_word2vec by doing (vocab_movie_shared - vocab_politeness)
(2) get vocab_movie_new by doing (vocab_movie_freq - vocab_politeness)
Note that the classifier doesn't recognize start and end tokens!
When doing classification, we should just remove start and end tokens, and then
classify
"""

freq_threshold = 23

all_tokens = [
    token
    for dataset in datasets
    for triple in dataset
    for turn in triple
    for token in turn]

fdist = FreqDist(all_tokens)
fdist_lst = fdist.most_common()
vocab_movie_freq = [
    token
    for (token, freq) in fdist_lst 
    if (freq >= freq_threshold)]

print(len(vocab_movie_freq))

num_UNKs = sum(
    [freq 
     for (_, freq) in fdist_lst
     if (freq < freq_threshold)])

print("Number of UNK tokens: %d" % num_UNKs)
print("Percentage of UNK tokens: %.1f" % (num_UNKs / len(all_tokens)))

vocab_word2vec = list(model.vocab)

# get the vocab that is out of politeness vocab
new_vocab_freq = list(
    set(vocab_movie_freq).difference(set(vocab_politeness)))

shared_vocab_movie = sorted(
    list(set(new_vocab_freq).intersection(set(vocab_word2vec))))
new_vocab_movie = sorted(
    list(set(new_vocab_freq).difference(set(vocab_word2vec))))

special_tokens = [
    "<polite>", "<neutral>", "<rude>", 
    "START_TOKEN", "END_TOKEN"]
new_vocab_movie.extend(special_tokens)

vocab_movie = shared_vocab_movie + new_vocab_movie
vocab_all = vocab_politeness + vocab_movie
assert len(set(vocab_all)) == len(vocab_all), "Duplicates in vocab_all!!"

# Note aside:
# "UNK_TOKEN" in vocab_politeness,
# The other 5 special tokens in new_vocab_movie

print("Politeness vocab size: %d" % len(vocab_politeness))
print("Newly added movie vocab size: %d" % len(vocab_movie))
print("Total vocab size: %d" % len(vocab_all))


# In[ ]:


"""
Obtain the reduced word2vec embedding matrix
"""
embedding_word2vec_movie = model[shared_vocab_movie]


# In[ ]:


"""
Create dictionaries between indices and tokens
"""
index2token = {i: token for (i, token) in enumerate(vocab_all)}
token2index = {token: i for (i, token) in enumerate(vocab_all)}


# In[ ]:


def replace_with_index(token, vocab, dictionary):
#     if token in vocab:
    try:
        return dictionary[token]
#     else:
    except:
        return dictionary["UNK_TOKEN"]


# In[ ]:


indexed_datasets = [
    [[[replace_with_index(token, vocab_all, token2index)
       for token in turn]
      for turn in lst]
     for lst in dataset]
    for dataset in truncated_datasets]


# In[ ]:


lsts = [
    vocab_all,
    shared_vocab_movie,
    new_vocab_movie,
    indexed_datasets[0][0],
    indexed_datasets[0][1],
    indexed_datasets[1][0],
    indexed_datasets[1][1],
    indexed_datasets[2][0],
    indexed_datasets[2][1],
    embedding_word2vec_movie
]

pickle_lst = [
    "vocab_all",
    "shared_vocab_movie",
    "new_vocab_movie",
    "movie_train_source",
    "movie_train_target",
    "movie_valid_source",
    "movie_valid_target",
    "movie_test_source",
    "movie_test_target",
    "embedding_word2vec_movie"
]

# data_path = "/usr/xtmp/tn9td/vocab/preprocessing/"
pickle_files = [
#     os.path.join(data_path, file + ".pkl")
    os.path.join(target_path, file + ".pkl")
    for file in pickle_lst]

dump_pickles(pickle_files, lsts)


# In[14]:


# vocab_all = load_pickle(target_path + "vocab_all.pkl")

# """
# Create dictionaries between indices and tokens
# """
# index2token = {i: token for (i, token) in enumerate(vocab_all)}
# token2index = {token: i for (i, token) in enumerate(vocab_all)}


# pkl = load_pickle(target_path + "movie_test_source.pkl")
# txt = read_lines(target_path + "source_test_text.txt")

# grouped_txt = group_lst(txt, 2)

# print(len(pkl))
# print(len(grouped_txt))

# assert(len(pkl) == len(grouped_txt))

# text_dict = {decode2string(index2token, p[:(-1)]): t # -1 to remove the last END_TOKEN
#              for (p, t) in zip(pkl, grouped_txt)}
# dump_pickle(target_path + "test_dict.pkl", text_dict)

