#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Don't forget to append end token for Subtle dataset!!
"""

# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os
import sys
from pprint import pprint
from pathlib import Path
from math import exp
import itertools
import string
from pprint import pprint
sys.path.extend([".", "../", "../.."])
from src.model.seq2seq_politeness import Seq2Seq
from src.model.util import gpu_config 
from src.basic.util import (shuffle, remove_duplicates, pad,
                            unzip_lst, zip_lsts,
                            prepend, append,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)
import argparse


# In[ ]:


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test continuous-LFT model")
    parser.add_argument(
        "--test", action="store_true",
        help="whether we are testing, default to False")
    parser.add_argument(
        "--ckpt_generator", type=str, default="ckpt/seq2seq_RL_pretrain_3",
        help="path to model files")
    args = parser.parse_args()
    return args

args = parse_args()
infer_only = args.test
force_restore_point = args.ckpt_generator


# In[2]:


start_epoch = 0
total_epochs = 40

debugging = False
pretrain = False
continuous_label = True
clip_loss = False
reorganize = False

gpu_start_index = 0
learning_rate = 0.001
monotonic_attention = True
output_layer = False

get_PPL = False
PPL_all = False

force_restore = True

"""
gpu configurations
"""
num_gpus = 1
batch_size = 96
assert batch_size % num_gpus == 0
batch_size_per_gpu = batch_size // num_gpus

extra_str = "_cont_LFT"

if monotonic_attention:
    extra_str += "_monotonic"
    ("Applying monotonic attention.")
    
if pretrain:
    extra_str += "_pretrain"
    
if debugging:
    batch_size = 4


# In[3]:


"""
Load pickled lists
"""
data_path = "data/MovieTriples/"
politeness_path = "data/Stanford_politeness_corpus/"
ckpt_path = "ckpt/"

"""
Load pickled lists
"""
filenames = [
    "vocab_all.pkl",
    "shared_vocab_politeness.pkl", "new_vocab_politeness.pkl",
    "shared_vocab_movie.pkl", "new_vocab_movie.pkl",
    "embedding_word2vec_politeness.pkl", "embedding_word2vec_movie.pkl",
    "movie_train_source.pkl", "movie_train_target.pkl",
    "movie_valid_source.pkl", "movie_valid_target.pkl",
    "movie_test_source.pkl", "movie_test_target.pkl",
    "polite_movie_target.pkl", "neutral_movie_target.pkl", "rude_movie_target.pkl"]
    
files = [
    os.path.join(politeness_path if "politeness" in filename else data_path, filename) 
    for filename in filenames]

# Load files
data = load_pickles(files)

vocab = data[0]
shared_vocab_politeness = data[1]
new_vocab_politeness = data[2]
shared_vocab_movie = data[3]
new_vocab_movie = data[4]
embedding_word2vec_politeness = data[5]
embedding_word2vec_movie = data[6]

source_train = data[7] 
target_train = data[8] 
source_valid = data[9]
target_valid = data[10]
source_test = data[11]
target_test = data[12]
triple_lsts = data[13:]


# In[4]:


def zip_remove_duplicates_unzip(lsts):
    zipped = zip_lsts(lsts)
    zipped_without_duplicates = remove_duplicates(zipped)    
    unzipped = unzip_lst(zipped_without_duplicates)
    return unzipped


# In[5]:


if not pretrain:
    [source_train, target_train] = zip_remove_duplicates_unzip([source_train, target_train])
    print(len(source_train))


# In[6]:


shared_vocab_size_politeness = len(shared_vocab_politeness)
shared_vocab_size_movie = len(shared_vocab_movie)

special_tokens = [
    "UNK_TOKEN", "START_TOKEN", "END_TOKEN",
    "<polite>", "<neutral>", "<rude>"]

vocab_size = len(vocab)

# Index vocabulary
index2token = {i: token for (i, token) in enumerate(vocab)}
token2index = {token: i for (i, token) in enumerate(vocab)}

[unk_token, start_token, end_token,
 polite_label, neutral_label, rude_label] = [token2index[token] 
                                             for token in special_tokens]

labels = [polite_label, neutral_label, rude_label]
num_labels = len(labels)

new_vocab_size_politeness = len(new_vocab_politeness)
new_vocab_size_movie = len(new_vocab_movie)
assert (1 + shared_vocab_size_politeness + new_vocab_size_politeness # +1 for "UNK"
        + shared_vocab_size_movie + new_vocab_size_movie) == vocab_size


# In[7]:


tags = ["<person>", "<number>", "<continued_utterance>"]
ner_tokens = [token2index[token] for token in tags]
unk_indices = [unk_token] + ner_tokens


# In[8]:


# LFT_data_file = "/playpen/home/tongn/Stanford_politeness_corpus/" + "LFT_continuous_label_train.pkl"
# LFT_data = load_pickle(LFT_data_file)
LFT_data = list(itertools.chain(*triple_lsts))


# In[9]:


if reorganize:
    words_path = "/home/tongn/politeness-generation/src/data/"
    polite_lst = set(
        [token2index[word] 
         for word in read_lines(words_path + "polite_words.txt")
         if word in vocab])
    rude_lst = set(
        [token2index[word] 
         for word in read_lines(words_path + "swear_words_wikitionary.txt")
         if word in vocab])
    
    LFT_examples = []
    counter = 0
    for (source, target, score) in LFT_data:
        if len(set(target).intersection(set(polite_lst))) > 0 and score < 0.8:
            LFT_examples.append([1.0, source, target])
            counter += 1
        elif len(set(target).intersection(set(rude_lst))) > 0 and score > 0.2:
            LFT_examples.append([0.0, source, target])
            counter += 1
        else:
            LFT_examples.append([source, target, score])
    print("Moved %d examples" % counter)
else:
    LFT_examples = LFT_data


# In[11]:


"""
Shared hyperparameters
"""
beam_width = 2
length_penalty_weight = 1.0
clipping_threshold = 5.0 # threshold for gradient clipping
embedding_size = 300

"""
seq2seq hyperparameters
"""
hidden_size = 512
num_layers = 2
max_iterations = 34
dropout_rate = 0.2
attention_size = 512
attention_layer_size = 256


# In[12]:


if debugging:
    if pretrain:
        data_dict = {
            "train": (source_train[:8], target_train[:8])}
    else:
        data_dict = {
            "train": (source_train[:8], target_train[:8]),
            "valid": (source_valid[:8], target_valid[:8]),
            "test": (source_test[:8], target_test[:8])}
else:
    if pretrain:
        data_dict = {
            "train": (source_train, target_train)}
    else:
        data_dict = {
            "train": (source_train, target_train),
            "valid": (source_valid, target_valid),
            "test": (source_test, target_test)}


# In[13]:


def build_model():
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        model = Seq2Seq(
            batch_size,
            shared_vocab_size_politeness, new_vocab_size_politeness,
            shared_vocab_size_movie, new_vocab_size_movie,
            embedding_word2vec_politeness, embedding_word2vec_movie, 
            embedding_size, hidden_size, num_layers,
            max_iterations,
            start_token, end_token, unk_indices,
            attention_size=attention_size, 
            attention_layer_size=attention_layer_size,
            beam_width=beam_width, length_penalty_weight=length_penalty_weight,
            gpu_start_index=gpu_start_index, 
            num_gpus=num_gpus,
            learning_rate=learning_rate, 
            clipping_threshold=clipping_threshold,
            monotonic_attention=monotonic_attention,
            continuous_label=continuous_label,
            output_layer=output_layer,
            clip_loss=clip_loss)
        saver_seq2seq = tf.train.Saver(var_list=model.trainable_variables)
        if start_epoch == 0:
            exclude_indices = [5]
            if monotonic_attention:
                exclude_indices.extend([37, 38, 39])
            restore_vars = [
                var
                for (i, var) in enumerate(model.trainable_variables)
                if i not in exclude_indices]
            saver_restore = tf.train.Saver(var_list=restore_vars)
        else:
            saver_restore = saver_seq2seq
    print("Done building model graph.")
    return (model, graph, saver_seq2seq, saver_restore)

(model, graph, saver_seq2seq, saver_restore) = build_model()


# In[14]:


# pprint(list(enumerate(model.trainable_variables)))
# input("pause")


# In[15]:


def run_seq2seq(sess, mode, epoch, feed_score=1.0):
    """see if we need to append end_token"""    
    is_training = (mode == "train")
    
    if is_training:
        (source_lst, target_lst, score_lst) = unzip_lst(LFT_examples)        
    else:
        (source_lst, target_lst) = data_dict[mode]
        score_lst = [feed_score] * len(source_lst)
    
#     source_lst = source_lst[:batch_size * 2]
#     target_lst = target_lst[:batch_size * 2]
#     score_lst = score_lst[:batch_size * 2]
    
    num_examples = len(source_lst)
    assert num_examples >= batch_size
    num_batches = num_examples // batch_size
    
    keep_prob = (1 - dropout_rate) if is_training else 1.0
    start_tokens = [start_token] * batch_size
    
    total_loss = 0.0
    num_tokens = 0
    zipped_lst = []
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        
        sources = source_lst[start:end]
        source_lengths = list(map(len, sources))
        targets = target_lst[start:end]
        target_lengths = list(map(len, targets))
        
        scores = score_lst[start:end]
        
        feed_dict = {
            model.source: pad(sources, source_lengths),
            model.source_length: source_lengths,
            model.target: pad(targets, target_lengths),
            model.target_length: target_lengths,
            model.start_tokens: start_tokens,
            model.keep_prob: keep_prob,
            model.is_training: is_training,
            model.score: scores}
        
        if is_training:
            fetches = [model.batch_total_loss, model.batch_num_tokens, model.apply_gradients_op]
        else:
            fetches = [model.batch_sample_ids_beam, model.batch_final_lengths_beam]
        
        result = sess.run(fetches, feed_dict=feed_dict)
        
        if is_training:
            total_loss += result[0]
            num_tokens += result[1]
            print("Epoch (%s) %d Batch %d perplexity: %.2f" % 
                  (mode, epoch, i, exp(result[0] / result[1])))
            print("Perplexity so far:", exp(total_loss / num_tokens))
        else:
            print("Finished testing batch %d" % i)
            responses = [response[:length] 
                         for (response, length) 
                         in zip(result[0].tolist(), result[1].tolist())]
            zipped = zip_lsts([sources, targets, responses])
            zipped_lst.extend(zipped)
                    
    if is_training:
        print("Epoch (%s) %d average perplexity: %.2f" % 
              (mode, epoch, exp(total_loss / num_tokens)))
        if not get_PPL:
            saver_seq2seq.save(sess, "%sseq2seq_RL%s_%d" % (ckpt_path, extra_str, epoch))
            print("Checkpoint saved for epoch %d." % epoch)
                    
    return zipped_lst


# In[16]:


config = gpu_config()

num_epochs = total_epochs - start_epoch
assert num_epochs >= 0

with tf.Session(graph=graph, config=config) as sess:
    sess.run(tf.global_variables_initializer())
    print("Initialized.")
    
    if force_restore or start_epoch > 0:
        if force_restore:
            restore_ckpt = force_restore_point
        else:
            restore_ckpt = "%sseq2seq_RL%s_%d" % (ckpt_path, extra_str, start_epoch - 1)
        
        saver_restore.restore(sess, restore_ckpt)
        print("Restored from", restore_ckpt)
        
    for i in xrange(num_epochs):
        if not infer_only:
            mode = "train"
            run_seq2seq(sess, mode, i + start_epoch)
            
            if pretrain:
                continue
            
            mode = "valid"
            score_range = [1.0]
            zipped = run_seq2seq(sess, mode, i + start_epoch, feed_score=score_range[0])
            
        if infer_only and not get_PPL and (i + start_epoch - 1) % 5 == 0: # for getting perplexity of test data, use train branch
            print("Inferring on test set...")
            mode = "test"

            responses_lst = []
            source_lst = []
            target_lst = []
            score_range = list(np.arange(0.0, 1.1, 0.5))
            for score in score_range:
                zipped_responses = run_seq2seq(
                    sess, mode, i + start_epoch, feed_score=score)
                (source_lst, target_lst, responses) = unzip_lst(zipped_responses)
                responses_lst.append(responses)
            num_responses = len(responses_lst[0])    

            zipped = zip_lsts([source_lst, target_lst] + responses_lst)
        
        flattened = [decode2string(index2token, sent, end_token=end_token, remove_END_TOKEN=True) 
                     for tp in zipped for sent in tp]

        # now we mark sentences that are generated by our model
        num_lines = len(score_range) + 2
        marked_G = [("G: " + sent)
                    if k % num_lines == 1 else sent
                    for (k, sent) in enumerate(flattened)]

        marked_M = [("M: " + sent) 
                    if k % num_lines in range(2, num_lines) else sent
                    for (k, sent) in enumerate(marked_G)]
        
        filename = ("%sseq2seq_RL_%s_result%s_%d.txt" % 
                    ("output/", mode, extra_str, i + start_epoch))

        write_lines(filename, marked_M)

        # only need 1 epoch for inferring or getting PPL
        if infer_only or get_PPL: 
            break


# In[ ]:




