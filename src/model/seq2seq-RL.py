#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os
from nltk import FreqDist
from numpy.random import choice
import sys
from pathlib import Path
from math import exp
import itertools
import string
import argparse

import sys
sys.path.extend([".", "../", "../.."])
from src.basic.util import (shuffle, remove_duplicates, zip_remove_duplicates_unzip, 
                            unzip_lst, zip_lsts,
                            prepend, append, avg,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)
from src.model.util import (concat_states, get_keep_prob, create_cell,
                            create_MultiRNNCell, lstm, create_placeholders, 
                            create_embedding, dynamic_lstm, bidirecitonal_dynamic_lstm, 
                            average_gradients, apply_grads, apply_multiple_grads, 
                            compute_grads, attention, 
                            get_bad_mask, get_unk_mask, get_sequence_mask,
                            calculate_BLEU, calculate_BLEUs, get_BLEUs, 
                            tile_single_cell_state, tile_multi_cell_state,
                            gpu_config, get_saver)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train/Val/Test seq2seq-RL politeness model")
    parser.add_argument(
        "--ckpt_generator", type=str, default="ckpt/seq2seq_RL_pretrain_3",
        help="path to model files")
    parser.add_argument(
        "--ckpt_classifier", type=str, default="ckpt/politeness_classifier_2",
        help="path to classifier checkpoints")
    parser.add_argument(
        "--test", action="store_true",
        help="whether we are testing, default to False")
    parser.add_argument(
        "--batch_size", type=int, default=384,
        help="batch size[384]")
    parser.add_argument(
        "--data_path", type=str, default="data/MovieTriples/",
        help="path to the indexed polite/rude/neutral utterances")
    parser.add_argument(
        "--politeness_path", type=str, default="data/Stanford_politeness_corpus/",
        help="path to the Stanford Politeness Corpus")
    parser.add_argument(
        "--pretrain", action="store_true",
        help="whether we perform pretraining with the SubTle corpus")
    args = parser.parse_args()
    return args

args = parse_args()
batch_size = args.batch_size
data_path = args.data_path
politeness_path = args.politeness_path
ckpt_generator = args.ckpt_generator
ckpt_classifier = args.ckpt_classifier
pretrain = args.pretrain

ckpt_path = "ckpt/"

start_epoch = 0
total_epochs = 4 if pretrain else 40

infer_only = args.test
get_PPL = False
fetch_embedding = False

"""
gpu configurations
"""
num_gpus = 1
gpu_start_index = 1
assert batch_size % num_gpus == 0
batch_size_per_gpu = batch_size // num_gpus

"""
flags
"""
no_unk = True
thresholding = True

polite_training = False if pretrain else True
flip_polite = False # set this on to train a very rude dialogue system
credit_assignment = False

assert not (flip_polite and not polite_training), "If flip_polite is True, then polite_training must be True"

"""
RL training weights
"""
beta = 2.0

# Extra strings to append to file name
extra_str = ""
if pretrain:
    extra_str += "_pretrain"
if polite_training:
    if credit_assignment:
        extra_str += "_credit"
        
    if flip_polite:
        extra_str += "_rude_%.1f" % beta
    else:
        extra_str += "_polite_%.1f" % beta


# In[3]:


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
    "movie_test_source.pkl", "movie_test_target.pkl"]

if pretrain:
    filenames.extend(["Subtle_source.pkl", "Subtle_target.pkl"])
    
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
source_test = data[11]
target_test = data[12]
if get_PPL:    
    source_train = source_test
    target_train = target_test
else:
    if pretrain:
        source_train = data[13]
        target_train = data[14]
    else:
        source_train = data[7] + data[9]
        target_train = data[8] + data[10]

# Debugging only
# source_train = source_train[:512]
# target_train = target_train[:512]
        
if infer_only:
    [source_test, target_test] = zip_remove_duplicates_unzip([source_test, target_test])
else:
    [source_train, target_train] = zip_remove_duplicates_unzip([source_train, target_train])

print(len(source_train))

# Also eliminate empty target/source, and list of lists
no_empty = [
    [source, target] for (source, target) 
    in zip(source_train, target_train)
    if (source != [] and target != []
        and not isinstance(source[0], list) 
        and not isinstance(target[0], list))]
[source_train, target_train] = unzip_lst(no_empty)

print(len(source_train))

if pretrain:
    source_threshold = 32
else:
    source_threshold = 32 * 2
    
zipped_lst = [
    (source, target) 
    for (source, target) 
    in zip(source_train, target_train)
    if (len(source) <= source_threshold 
        and len(source) >= 5
        and len(target) <= 32)]
#         and source.count(0) == 0
#         and target.count(0) == 0
print(len(zipped_lst))

[source_train, target_train] = unzip_lst(zipped_lst)

def pad_tensor(tensor, lengths):
    max_length = tf.reduce_max(lengths)
    padded = tf.pad(
        tensor, 
        [[0, 0], 
         [0, max_iterations - max_length]])
    return padded

def decode(cell, helper, initial_state):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state)
    (decoder_outputs, _, final_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True,
        maximum_iterations=max_iterations, swap_memory=True)
    return (decoder_outputs, final_lengths)

def get_valid_mask(inputs):
    valid_mask = tf.cast(
        tf.logical_and(
            tf.not_equal(inputs, 0),
            tf.not_equal(inputs, end_token)),
        tf.float32)
    return valid_mask

def get_prob_dist(lst):
    fdist = FreqDist(lst)
    fdist_lst = fdist.most_common()
    [vals, freqs] = unzip_lst(fdist_lst)
    freq_total = sum(freqs)
    probs = [freq / freq_total for freq in freqs]
    return (vals, probs)

def generate_samples(vals, num_samples, probs=None):
    return list(choice(vals, num_samples, probs))

source_lengths = [len(source) for source in source_train]
[lengths, length_probs] = get_prob_dist(source_lengths)

all_tokens = [
    token 
    for lst in [source_train, target_train]
    for sent in lst
    for token in sent]
[tokens, token_probs] = get_prob_dist(all_tokens)

batch_sampled_lengths = generate_samples(lengths, batch_size, length_probs)
batch_sampled_source = [
    generate_samples(tokens, length, token_probs)
    for length in batch_sampled_lengths]

shared_vocab_size_politeness = len(shared_vocab_politeness)
shared_vocab_size_movie = len(shared_vocab_movie)

special_tokens = ["UNK_TOKEN", "START_TOKEN", "END_TOKEN"]
vocab_size = len(vocab)

# Index vocabulary
index2token = {i: token for (i, token) in enumerate(vocab)}
token2index = {token: i for (i, token) in enumerate(vocab)}

[unk_token, start_token, end_token] = [token2index[token] for token in special_tokens]

new_vocab_size_politeness = len(new_vocab_politeness)
new_vocab_size_movie = len(new_vocab_movie)
assert (shared_vocab_size_politeness + new_vocab_size_politeness + 1
        + shared_vocab_size_movie + new_vocab_size_movie) == vocab_size

vocab_size_politeness = 1 + shared_vocab_size_politeness + new_vocab_size_politeness

tags = ["<person>", "<number>", "<continued_utterance>"]
ner_tokens = [token2index[token] for token in tags]
unk_indices = [unk_token, ner_tokens[2]]

def dictionary_lookups(lst, dictionary):
    converted = [dictionary[x] for x in lst]
    return converted

"""
Shared hyperparameters
"""
beam_width = 2
length_penalty_weight = 1.0
clipping_threshold = 5.0 # threshold for gradient clipping
embedding_size = 300
learning_rate = 0.001

"""
seq2seq hyperparameters
"""
hidden_size_encoder = 256
hidden_size_decoder = 512
num_layers_encoder = 4
num_layers_decoder = num_layers_encoder // 2
if pretrain:
    max_iterations = 40 + 2 # should be computed by 95 percentile of all sequence lengths
                            # +2 for <start> and <end>
else:
    max_iterations = 32 + 2
    
dropout_rate = 0.2
attention_size = 512
attention_layer_size = 256
start_tokens = [start_token] * batch_size

"""
RL training parameters
"""
threshold = 0.2
baseline = 0.5 # since our training data is balanced, 0.5 is reasonable 

"""
classifier hyperparameters
"""
hidden_size_classifier = 256
num_classes = 2
filter_sizes = [3, 4, 5]
num_filters = 75

def get_mask(seqs, indices):
    tensor = tf.convert_to_tensor(indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(tensor, [len(indices), 1, 1]))
    mask = tf.reduce_any(bool_matrix, axis=0)
    return mask

"""
Gather indices from an arbitrary dimension
"""
def gather_axis(params, indices, axis=0):
    logits_lst = tf.unstack(params, axis=axis)
    gathered_logits_lst = [logits_lst[i]
                           for i in indices]    
    gathered_logits = tf.stack(gathered_logits_lst, axis=axis)
    return gathered_logits

def filter_with_threshold(score):
    filtered_score = tf.cond(
        tf.logical_or(score <= threshold, score >= 1 - threshold),
        lambda: score, lambda: baseline)
    return filtered_score

"""
Mask out invalid positions of 'tensor' and compute its softmax.
This way the invalid positions will have an output of zero!
"""
def softmax_with_mask(tensor, mask):
    exp = tf.exp(tensor) * mask
    sum_exp = tf.reduce_sum(exp)
    softmax = tf.cond(
        sum_exp > 1.0, # if there is at least one valid token (avoid divide by 0)
        lambda: exp / sum_exp,
        lambda: tf.zeros_like(tensor, dtype=tf.float32))
    
    softmax_renorm_by_length = softmax * tf.reduce_sum(mask)

    return softmax_renorm_by_length

def build_classifier(inputs, seq_lengths, reuse):
    
    max_seq_length = tf.reduce_max(seq_lengths)
    keep_prob = tf.convert_to_tensor(1.0) # since we are doing inference only
    
    # Get the mask of all valid tokens (Assuming that unk_token == pad_token == 0)
    valid_mask = get_valid_mask(inputs)
    
    # Embedding layer
    with tf.variable_scope("embedding", reuse=reuse):
        embedding_unk = tf.get_variable(
            "embedding_unk",
            shape=[1, embedding_size])
        embedding_politeness_original = tf.get_variable(
            "embedding_politeness_original",
            shape=[shared_vocab_size_politeness, embedding_size])
        embedding_politeness_new = tf.get_variable(
            "embedding_politeness_new",
            shape=[new_vocab_size_politeness, embedding_size])
        embeddings = [
            embedding_unk, embedding_politeness_original, 
            embedding_politeness_new]            
        embedding = tf.concat(embeddings, axis=0)
        embedded_inputs = tf.nn.embedding_lookup(
            embedding, inputs)            

    with tf.variable_scope("lstm", reuse=reuse):
        cell_fw = lstm(
            embedding_size, hidden_size_classifier,
            keep_prob, reuse)
        cell_bw = lstm(
            embedding_size, hidden_size_classifier,
            keep_prob, reuse)

        (outputs, final_state) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, embedded_inputs,
            sequence_length=seq_lengths,
            dtype=tf.float32,
            swap_memory=True)

    # H's shape: batch_size_per_gpu * max_seq_length * (2 * hidden_size_classifier)
    H = tf.concat(outputs, axis=2)

    # CNN + maxpooling layer
    with tf.variable_scope("CNN_maxpooling", reuse=reuse):
        H_expanded = tf.expand_dims(H, axis=-1) # expand H to 4-D (add a chanel dim) for CNN
        pooled_outputs = []

        for (j, filter_size) in enumerate(filter_sizes):
            with tf.variable_scope("filter_%d" % j): # sub-scope inherits the reuse flag
                # CNN layer
                filter_shape = [filter_size, 2 * hidden_size_classifier, 1, num_filters]
                W_conv = tf.get_variable(
                    "W_conv", shape=filter_shape,
                    initializer=tf.contrib.layers.xavier_initializer_conv2d())
                b_conv = tf.get_variable(
                    "b_conv", shape=[num_filters],
                    initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(
                    H_expanded, W_conv, strides=[1, 1, 1, 1], padding="VALID")
                output_conv = tf.nn.relu(tf.nn.bias_add(conv, b_conv))
                # maxpooling layer
                maxpooled_lst = [] # maxpooled results for each example in the batch
                for k in range(batch_size_per_gpu):
                    sequence_conv = output_conv[
                        k, :seq_lengths[k], 0, :]
                    maxpooled = tf.reduce_max(sequence_conv, axis=0)
                    maxpooled_lst.append(maxpooled)
                batch_maxpooled = tf.stack(maxpooled_lst, axis=0)
                pooled_outputs.append(batch_maxpooled)
        h_maxpool = tf.concat(pooled_outputs, axis=1)
        h_maxpool_dropout = tf.nn.dropout(h_maxpool, keep_prob=keep_prob)

    # output layer (fully connected)
    with tf.variable_scope("output", reuse=reuse):           
        num_filters_total = num_filters * len(filter_sizes)
        W_out = tf.get_variable(
            "W_out", shape=[num_filters_total, num_classes],
            initializer=tf.contrib.layers.xavier_initializer())
        b_out = tf.get_variable(
            "b_out", shape=[num_classes],
            initializer=tf.constant_initializer(0.1))
        logits = tf.nn.xw_plus_b(
            h_maxpool_dropout, weights=W_out, biases=b_out)
        scores = tf.nn.softmax(logits, axis=-1)
        politeness_scores = scores[:, 1]

    optimizer = tf.train.AdamOptimizer(0.001)
    
    if credit_assignment:
        # Note: currently the tf.gather has to appear after computing gradients,
        # otherwise tf.gradients returns None!
        credit_weights_lst = []
        for j in range(batch_size_per_gpu):
    #         embedding_grads = tf.convert_to_tensor(
    #             tf.gradients(politeness_scores[j], embedding)[0])

            [embedding_grads, _] = unzip_lst(
                optimizer.compute_gradients(politeness_scores[j], var_list=embeddings))
            
            # Display warning message if some element in embedding_grads is "None"
            for grad in embedding_grads:
                if grad is None:
                    print("Warning: one of the credit assignment embedding grads is None: ", grad)
            
            embedding_grads_concat = tf.concat(embedding_grads, axis=0)
            gathered_embedding_grads = tf.gather(
                embedding_grads_concat, inputs[j, :])
            normed_embedding_grads = tf.norm(gathered_embedding_grads, axis=1)         
            credit_weights = softmax_with_mask(normed_embedding_grads, valid_mask[j, :])
            credit_weights_lst.append(credit_weights)

        stacked_credit_weigths = tf.stack(credit_weights_lst, axis=0)
    else:
        stacked_credit_weigths = tf.zeros_like(inputs)

    return (politeness_scores, stacked_credit_weigths)

def pad_and_truncate(sample_ids, lengths):
    max_length = tf.reduce_max(lengths)
    padded_sample_ids = tf.pad(
         sample_ids, 
         [[0, 0], 
          [0, max_iterations - max_length]])
    truncated_sample_ids = padded_sample_ids[:, :max_iterations] # truncate length
    return truncated_sample_ids

def build_seq2seq(input_seqs, target_seqs, filtered_target_seqs,
                  input_seq_lengths, target_seq_lengths, 
                  is_training):

    with tf.variable_scope("seq2seq"):
        with tf.device('/cpu:0'):            
            reuse = False
            
            if get_PPL:
                keep_prob = tf.convert_to_tensor(1.0)
            else:
                keep_prob = get_keep_prob(dropout_rate, is_training)
                        
            sequence_mask = get_sequence_mask(target_seq_lengths)
    
            unk_mask = get_mask(target_seqs, unk_indices)
            decoder_mask = tf.logical_and(
                sequence_mask, tf.logical_not(unk_mask))
            decoder_mask_float = tf.cast(decoder_mask, tf.float32)
            
            # Embed inputs
            with tf.variable_scope("embedding"):
                embedding = create_embedding(
                    embedding_word2vec_politeness, embedding_word2vec_movie,
                    shared_vocab_size_politeness, shared_vocab_size_movie,
                    new_vocab_size_politeness, new_vocab_size_movie, 
                    "seq2seq")
                embedded_input_seqs = tf.nn.embedding_lookup(
                    embedding, input_seqs)
                embedded_target_seqs = tf.nn.embedding_lookup(
                    embedding, target_seqs)
            
            # Optimizer
            optimizer = tf.train.AdamOptimizer(learning_rate)
            
            tower_grads = []
            if credit_assignment:
                tower_grads_polite = []
            sample_ids_lst = []
            final_lengths_lst = []
            sampled_sample_ids_lst = []
            sampled_final_lengths_lst = []
            reuse = False
            trainable_variables = []
            
            num_tokens_lst = []
            total_losses = []
    for i in xrange(num_gpus):
        with tf.device("/gpu:%d" % (gpu_start_index + i)):
            with tf.variable_scope("seq2seq"):
                if (i == 1):
                    reuse = True

                start = i * batch_size_per_gpu
                end = start + batch_size_per_gpu
                
                input_max_seq_length = tf.reduce_max(input_seq_lengths[start:end])
                target_max_seq_length = tf.reduce_max(target_seq_lengths[start:end])
    
                with tf.variable_scope("encoder", reuse=reuse):
                    cell_fw = create_MultiRNNCell( 
                        [hidden_size_encoder] * (num_layers_encoder // 2),
                        keep_prob, num_proj=None, reuse=reuse)
                    cell_bw = create_MultiRNNCell( 
                        [hidden_size_encoder] * (num_layers_encoder // 2),
                        keep_prob, num_proj=None, reuse=reuse)
                    (encoder_outputs_original, encoder_final_state_original) = bidirecitonal_dynamic_lstm(
                        cell_fw, cell_bw, 
                        embedded_input_seqs[start:end, :input_max_seq_length, :],
                        input_seq_lengths[start:end])
                
                    [encoder_outputs, encoder_seq_lengths, encoder_final_state] = tf.cond(
                        is_training,
                        lambda: [encoder_outputs_original, 
                                 input_seq_lengths[start:end],
                                 encoder_final_state_original],
                        lambda: [tf.contrib.seq2seq.tile_batch(encoder_outputs_original, beam_width),
                                 tf.contrib.seq2seq.tile_batch(input_seq_lengths[start:end], beam_width),
                                 tile_multi_cell_state(encoder_final_state_original)]) # only works for decoder that has >1 layers!
                
                with tf.variable_scope("decoder", reuse=reuse):
                    decoder_cell = create_MultiRNNCell(
                        [hidden_size_decoder] * (num_layers_decoder),
                        keep_prob, num_proj=vocab_size,
                        memory=encoder_outputs,
                        memory_seq_lengths=encoder_seq_lengths,
                        reuse=reuse)

                    decoder_zero_state = tf.cond(
                        is_training,
                        lambda: decoder_cell.zero_state(batch_size_per_gpu, tf.float32),
                        lambda: decoder_cell.zero_state(
                            batch_size_per_gpu * beam_width, tf.float32))
                    
                    state_last = decoder_zero_state[-1].clone(
                        cell_state=encoder_final_state[-1])
                    state_previous = encoder_final_state[:-1]
                    decoder_initial_state = state_previous + (state_last, ) # concat tuples
                                            
                    # training helper (for teacher forcing)
                    helper_train = tf.contrib.seq2seq.TrainingHelper(
                        embedded_target_seqs[
                            start:end, :target_max_seq_length - 1, :], # get rid of end_token
                        target_seq_lengths[start:end] - 1) # the length is thus decreased by 1
                    
                    (decoder_outputs_train, _) = decode(
                        decoder_cell, helper_train,
                        initial_state=decoder_initial_state)
                    (logits, _) = decoder_outputs_train
                    
                    # Get trainable_variables 
                    # (up to now we already have all the seq2seq trainable vars)
                    if trainable_variables == []:
                        trainable_variables = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES, 
                            scope="seq2seq")
                    
                    loss_ML = tf.contrib.seq2seq.sequence_loss(
                        logits,
                        target_seqs[start:end, 1:target_max_seq_length], # get rid of start_token
                        decoder_mask_float[start:end, 1:target_max_seq_length])
                    num_tokens = tf.reduce_sum(
                        decoder_mask_float[start:end, 1:target_max_seq_length])
                        
                    num_tokens_lst.append(num_tokens)
                    
                    total_loss = loss_ML * num_tokens
                    total_losses.append(total_loss)
                    
                    if polite_training:
                        helper_sample = tf.contrib.seq2seq.SampleEmbeddingHelper(
                            embedding, start_tokens[start:end], end_token)
                        (decoder_outputs_sample, final_lengths_sample) = decode(
                            decoder_cell, helper_sample, decoder_initial_state)               
                        (logits_sample, sample_ids_sample) = decoder_outputs_sample
                        max_final_lengths_sample = tf.reduce_max(final_lengths_sample)
                        sampled_sample_ids_lst.append(
                            pad_and_truncate(sample_ids_sample, final_lengths_sample))
                        sampled_final_lengths_lst.append(final_lengths_sample)
                        
                        # Compute sampled sequence loss WITHOUT averaging (will do that later)
                        decoder_mask_sample = get_sequence_mask(final_lengths_sample, dtype=tf.float32)
                        seq_losses_sample = tf.contrib.seq2seq.sequence_loss(
                            logits_sample, sample_ids_sample,
                            decoder_mask_sample,
                            average_across_timesteps=False,
                            average_across_batch=False)
            
            if polite_training:
                with tf.variable_scope("classifier"): # jump back to the classifier scope
                    # Filter out tokens that the classifier doesn't know
                    vocab_mask = tf.cast(
                        sample_ids_sample < vocab_size_politeness,
                        tf.int32)
                    sample_ids_sample_classifier = sample_ids_sample * vocab_mask
                    
                    # Feed sampled ids to classifier
                    (scores_RL, credit_weights_RL) = build_classifier(
                        sample_ids_sample_classifier, final_lengths_sample, reuse)

                    # Stop gradients from propagating back
                    scores_RL_stop = tf.stop_gradient(scores_RL)
                    credit_weights_RL_stop = tf.stop_gradient(credit_weights_RL)
                    
                    if thresholding:
                        # Filter scores that are >= threshold and <= 1 - threshold
                        filtered_scores_RL = tf.map_fn(filter_with_threshold, scores_RL_stop)
                    else:
                        filtered_scores_RL = scores_RL_stop

                with tf.variable_scope("seq2seq"):
                    with tf.variable_scope("decoder", reuse=reuse):
                        # Get valid mask for sampled sequence
                        decoder_mask_classifier = tf.cast(
                            tf.not_equal(sample_ids_sample, 0),
                            tf.float32) # propagate back the whole sentence (including <end>)
                        
                        tiled_scores = tf.tile( # tile scores to 2D
                            tf.expand_dims(filtered_scores_RL - baseline, axis=1),
                            [1, max_final_lengths_sample])
                        
                        if flip_polite: # if we actually want a rude dialogue system
                            tiled_scores = -1.0 * tiled_scores

                        # Compute seq losses for polite-RL
                        seq_losses_classifier = (beta * seq_losses_sample 
                                                 * decoder_mask_classifier 
                                                 / tf.reduce_sum(decoder_mask_classifier)
                                                 * tiled_scores)
                        
                        if credit_assignment:
                            grads_polite = tf.gradients(
                                seq_losses_classifier, trainable_variables,
                                grad_ys=credit_weights_RL_stop) # credit weights as initial gradients
                            grads_polite = zip_lsts([grads_polite, trainable_variables])
                            tower_grads_polite.append(grads_polite)
                        else:
                            loss_polite = tf.reduce_sum(seq_losses_classifier)
            else:
                credit_weights_RL_stop = None

            with tf.variable_scope("seq2seq"):
                with tf.variable_scope("decoder", reuse=reuse):
                    # Infer branch (beam search!)
                    beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        decoder_cell, embedding, 
                        start_tokens[start:end], end_token,
                        decoder_initial_state, beam_width,
                        length_penalty_weight=length_penalty_weight)
                    output_beam = tf.contrib.seq2seq.dynamic_decode(
                        beam_search_decoder, 
    #                     impute_finished=True, # cannot be used with Beamsearch
                        maximum_iterations=max_iterations, 
                        swap_memory=True)
                    sample_ids = output_beam[0].predicted_ids[:, :, 0]
                    final_lengths = output_beam[2][:, 0]
                    
                    sample_ids_lst.append(
                        pad_and_truncate(sample_ids, final_lengths))
                    final_lengths_lst.append(final_lengths)
                    
        with tf.device("/gpu:%d" % (gpu_start_index + i)):
            with tf.variable_scope("seq2seq", reuse=reuse):
                # Compute loss
                loss = loss_ML

                if polite_training and not credit_assignment:
                    loss = loss + loss_polite
                
                # Compute tower gradients
                grads = compute_grads(loss, optimizer, trainable_variables)
                tower_grads.append(grads)
    
    with tf.device('/cpu:0'):
        with tf.variable_scope("seq2seq"):
            # Concat sample ids and their respective lengths
            batch_sample_ids = tf.concat(sample_ids_lst, axis=0)
            batch_final_lengths = tf.concat(final_lengths_lst, axis=0)

            if polite_training:
                batch_sampled_sample_ids = tf.concat(
                    sampled_sample_ids_lst, axis=0)
            
            batch_total_loss = tf.add_n(total_losses)
            batch_num_tokens = tf.add_n(num_tokens_lst)
                        
            # Thus, the effective batch size is actually batch_size_per_gpu
            if polite_training and credit_assignment:
                apply_gradients_op = apply_multiple_grads(optimizer, [tower_grads, tower_grads_polite])
            else:
                apply_gradients_op = apply_grads(optimizer, tower_grads)
            
                
    return (batch_sample_ids, batch_final_lengths, 
            batch_total_loss, batch_num_tokens,
            apply_gradients_op, credit_weights_RL_stop,
            embedding)

graph = tf.Graph()
with graph.as_default():
    with tf.device('/cpu:0'):
        # Create all placeholders
        (input_seqs, input_seq_lengths, 
         target_seqs, target_seq_lengths,
         is_training) = create_placeholders(batch_size)
    
        filtered_target_seqs = tf.placeholder(
            tf.int32, shape=[batch_size, None], 
            name="filtered_target_seqs")

    (batch_sample_ids, batch_final_lengths, 
     batch_total_loss, batch_num_tokens,
     apply_gradients_op, credit_weights_RL,
     embedding_seq2seq) = build_seq2seq(
        input_seqs, target_seqs, 
        filtered_target_seqs,
        input_seq_lengths, target_seq_lengths, 
        is_training)

with graph.as_default():    
    init = tf.global_variables_initializer()
    
    if polite_training:
        saver_classifier = tf.train.Saver(
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="classifier"))
    saver_seq2seq = tf.train.Saver(
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq2seq"))

"""
Pad a batch to max_sequence_length along the second dimension
Args:
    • input_seqs: a list of sequences
    • sequence_lengths: a list of sequence length
Returns:
    • padded
"""

def pad(input_seqs, sequence_lengths):
    max_length = max(sequence_lengths)
    padded = [input_seq + [0] * (max_length - sequence_length) 
              for (input_seq, sequence_length)
              in zip(input_seqs, sequence_lengths)]
    return padded

def run_seq2seq(sess, source_lst, target_lst, mode, epoch):
    assert len(source_lst) == len(target_lst)
    
    # End token is already there
    target_lst = prepend(target_lst, start_token)
    if pretrain: # Subtle does not contain <end>
        target_lst = append(target_lst, end_token)
    
    if (mode == "train"):
        training_flag = True
    else:
        training_flag = False

    num_batches = len(source_lst) // batch_size
        
    print("Number of batches:", num_batches)
    
    responses = []
    diff_lst = []
    total_loss = 0.0
    total_num_tokens = 0.0
    
    num_batches = 2
    for i in xrange(num_batches):
        
        start = batch_size * i
        end = start + batch_size
        
        source = source_lst[start:end]
        source_lengths = [len(seq) for seq in source]
        padded_source = pad(source, source_lengths)
        
        target = target_lst[start:end]
        target_lengths = [len(seq) for seq in target]
        padded_target = pad(target, target_lengths)

        feed_dict = {
            input_seqs: padded_source,
            input_seq_lengths: source_lengths,
            target_seqs: padded_target,
            target_seq_lengths: target_lengths,
            is_training: training_flag}
                
        if mode == "train":
            # if we only need PPL, then no need to back-prop
            if get_PPL:
                fetches = [batch_total_loss, batch_num_tokens]
            else:
                fetches = [batch_total_loss, batch_num_tokens, apply_gradients_op]
            
            result = sess.run(fetches, feed_dict=feed_dict)
            average_log_perplexity = result[0] / result[1]
            total_loss += result[0]
            total_num_tokens += result[1]
            print("Epoch (%s) %d Batch %d perplexity: %.2f" % 
                  (mode, epoch, i, exp(average_log_perplexity)))
            print("Perplexity so far:", exp(total_loss / total_num_tokens))
                        
        else:
            (ids, lengths) = sess.run(
                [batch_sample_ids, batch_final_lengths],
                feed_dict=feed_dict)

            batch_responses = [
                [index for index in response[:length]]
                for (response, length) 
                in zip(ids.tolist(), lengths.tolist())]
            responses.extend(batch_responses)
            print(f"Finished testing batch {i}.")

    if mode == "train":
        epoch_perplexity = total_loss / total_num_tokens
        print("Epoch (%s) %d average perplexity: %.2f" % 
              (mode, epoch, exp(epoch_perplexity)))
        
        if not get_PPL:
#             saver_seq2seq.save(sess, "%sseq2seq_RL%s_%d" % (data_path, extra_str, epoch))
            saver_seq2seq.save(sess, "%sseq2seq_RL%s_%d" % (ckpt_path, extra_str, epoch))
            print("Checkpoint saved for epoch %d." % epoch)
    else:
        return responses

num_epochs = total_epochs - start_epoch
assert num_epochs >= 0

config = gpu_config()    
with tf.Session(graph=graph, config=config) as sess:
    sess.run(init)
    print("Initialized.")
    
    if (not pretrain) or infer_only: # for pretraining we don't have anything to restore from
        saver_seq2seq.restore(sess, ckpt_generator)
        print("Restored from", ckpt_generator)
    
    # if infer only, we really don't need to restore anything
    if polite_training and not infer_only:
        saver_classifier.restore(sess, ckpt_classifier)
        print("Restored all varaibles from classifier.")
    
    for i in xrange(num_epochs):
        if not infer_only:
            run_seq2seq(sess, source_train, target_train, "train", i + start_epoch)
            (source_train, target_train) = shuffle(source_train, target_train)
    
        if (((i + start_epoch + 1) >= 10 # only test for later epochs
             and (i + start_epoch + 1) % 5 == 0)
            or infer_only
            and not get_PPL): # for getting perplexity of test data, use train branch
            responses = run_seq2seq(
                sess, source_test, target_test, "test", i + start_epoch)

#             # need to store all inferred responses in a pickle file
#             if infer_only:
#                 dump_pickle(
#                     "%sseq2seq_RL_result%s_%d_infer.pkl" % (data_path, extra_str, i + start_epoch), 
#                     responses)

            num_responses = len(responses)
            zipped = zip_lsts(
                [source_test[:num_responses], 
                 target_test[:num_responses],
                 responses])
            flattened = [decode2string(index2token, sent, remove_END_TOKEN=True) 
                         for tp in zipped for sent in tp]

            # now we mark sentences that are generated by our model
            marked_G = [("G: " + sent) 
                        if k % 3 == 1 else sent
                        for (k, sent) in enumerate(flattened)]

            marked_M = [("M: " + sent) 
                        if k % 3 == 2 else sent
                        for (k, sent) in enumerate(marked_G)]

            filename = "%sseq2seq_RL_result%s_%d.txt" % (data_path, extra_str, i + start_epoch)

            write_lines(filename, marked_M)

        # only need 1 epoch for inferring or getting PPL
        if infer_only or get_PPL: 
            break              

