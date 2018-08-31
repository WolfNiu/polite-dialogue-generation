# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import numpy as np
import tensorflow as tf
import os
import sys
from pathlib import Path
from math import exp
import itertools
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import string
import argparse

import sys
sys.path.append("../..")
from src.basic.util import (shuffle, remove_duplicates, zip_remove_duplicates_unzip,
                            unzip_lst, zip_lsts, split_triple_lst,
                            prepend, append, avg,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)
from src.model.util import (concat_states, get_keep_prob, dropout create_cell,
                            create_MultiRNNCell, lstm, create_placeholders, 
                            create_embedding, dynamic_lstm, bidirecitonal_dynamic_lstm, 
                            average_gradients, apply_grads, apply_multiple_grads, 
                            compute_grads, attention, decode, 
                            get_bad_mask, get_unk_mask, get_sequence_mask,
                            calculate_BLEU, calculate_BLEUs, get_BLEUs, 
                            get_valid_mask, pad_tensor, tile_single_cell_state, 
                            tile_multi_cell_state, gpu_config, get_saver)

def parse_args():
    parser.add_argument(
        "--ckpt_generator", type=str, default="../checkpoint/",
        help="path to generator files")
    parser.add_argument(
        "--ckpt_classifier", type=str, default="../checkpoint/",
        help="path to classifier files")
    parser.add_argument(
        "--test", action="store_true", 
        help="whether we are testing, default to False")
    parser.add_argument(
        "--batch_size", type=int, default=96, 
        help="batch size[96]")
    parser.add_argument(
        "--data_path", type=str, default="../data/",
        help="path to the indexed polite/rude/neutral utterances")
    args = parser.parse_args()
    return args

args = parse_args()
data_path = args.data_path
restore_path = args.ckpt_model

start_epoch = 0
total_epochs = 40

reorganize = False


"""
gpu configurations
"""
batch_size = 96
num_gpus = 1
assert batch_size % num_gpus == 0
batch_size_per_gpu = batch_size // num_gpus


"""
Load pickled lists
"""
filenames = [
    "vocab_all.pkl",
    "shared_vocab_politeness.pkl", "shared_vocab_movie.pkl",
    "new_vocab_politeness.pkl", "new_vocab_movie.pkl",
    "embedding_word2vec_politeness.pkl", "embedding_word2vec_movie.pkl",
    "movie_train_source.pkl", "movie_train_target.pkl",
    "movie_valid_source.pkl", "movie_valid_target.pkl",
    "movie_test_source.pkl", "movie_test_target.pkl",
    "polite_movie_target.pkl",  "neutral_movie_target.pkl", "rude_movie_target.pkl"]

files = [os.path.join(data_path, filename) for filename in filenames]

# Load files
data = load_pickles(files)

vocab = data[0]
shared_vocab_politeness = data[1]
shared_vocab_movie = data[2]
new_vocab_politeness = data[3]
new_vocab_movie = data[4]
embedding_word2vec_politeness = data[5]
embedding_word2vec_movie = data[6]
source_train = data[7] + data[9]
target_train = data[8] + data[10]
source_test = data[11]
target_test = data[12]
triple_lsts = data[13:]

[source_train, target_train] = zip_remove_duplicates_unzip([source_train, target_train])

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

tags = ["<person>", "<number>", "<continued_utterance>"]

ner_tokens = [token2index[token] for token in tags]
unk_indices = [unk_token, ner_tokens[2]]

bad_indices = []
good_indices = []

print([len(triple_lst) for triple_lst in triple_lsts])

if reorganize:
    (polite_kept_lst, polite_popped_lst) = split_triple_lst(triple_lsts[0], bad_indices)
    (rude_kept_lst, rude_popped_lst) = split_triple_lst(triple_lsts[2], good_indices)
    triple_lsts[0] = polite_kept_lst + rude_popped_lst
    triple_lsts[2] = rude_kept_lst + polite_popped_lst

    (neutral_kept_lst_polite, neutral_popped_lst_polite) = split_triple_lst(triple_lsts[1], good_indices)
    triple_lsts[1] = neutral_kept_lst_polite
    triple_lsts[0] = triple_lsts[0] + neutral_popped_lst_polite

    (neutral_kept_lst_rude, neutral_popped_lst_rude) = split_triple_lst(triple_lsts[1], bad_indices)
    triple_lsts[1] = neutral_kept_lst_rude
    triple_lsts[2] = triple_lsts[2] + neutral_popped_lst_rude
    
    print([len(triple_lst) for triple_lst in triple_lsts])


if reorganize:
    # Store reorganized utterances
    for (filename, triple_lst) in zip(filenames[13:], triple_lsts):
        dump_pickle(data_path + filename, triple_lst)

unzipped_triples = [unzip_lst(triple_lst) for triple_lst in triple_lsts]
[[polite_sources, polite_targets, _],
 [neutral_sources, neutral_targets, _],
 [rude_sources, rude_targets, _]] = unzipped_triples

sources_lst = [polite_sources, neutral_sources, rude_sources]
targets_lst = [polite_targets, neutral_targets, rude_targets]

labeled_sources_lst = [prepend(sources, label) 
                      for (sources, label) 
                      in zip(sources_lst, labels)]

# Comine the three parts to make train dataset
labeled_source_train = labeled_sources_lst[0] + labeled_sources_lst[1] + labeled_sources_lst[2]
labeled_target_train = polite_targets + neutral_targets + rude_targets
[labeled_source_train, labeled_target_train] = shuffle(labeled_source_train, labeled_target_train)

# Prepend labels for test dataset
source_test_polite = prepend(source_test, polite_label)
source_test_neutral = prepend(source_test, neutral_label)
source_test_rude = prepend(source_test, rude_label)


"""
Shared hyperparameters
"""
beam_width = 10
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
max_iterations = 32 + 2 # should be computed by 95 percentile of all sequence lengths
                        # +2 for <start> and <end>
dropout_rate = 0.2
attention_size = 512
attention_layer_size = 256
start_tokens = [start_token] * batch_size


def build_seq2seq(input_seqs, target_seqs, 
                  input_seq_lengths, target_seq_lengths, 
                  is_training):

    with tf.variable_scope("seq2seq"):
        with tf.device('/cpu:0'):            
            keep_prob = get_keep_prob(dropout_rate, is_training)
            
            # A mask that has 0's where it starts with <polite>
            # and 1's otherwise
            not_polite_mask = tf.tile(
                tf.expand_dims(
                    tf.not_equal(input_seqs[:, 0], polite_label),
                    axis=1),
                [1, tf.reduce_max(target_seq_lengths)])
                
            # The mask makes sure the tokens are valid 
            # and not one of the bad words (only for <polite>)
            bad_mask = tf.logical_or(
                get_bad_mask(target_seqs), not_polite_mask)
            sequence_mask = tf.logical_and(
                get_sequence_mask(target_seq_lengths), 
                bad_mask)
    
            unk_mask = get_unk_mask(target_seqs)
            decoder_mask = tf.cast(
                tf.logical_and(sequence_mask, unk_mask),
                tf.float32)
            
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
            sample_ids_lst = []
            final_lengths_lst = []
            losses_ML = []       
            reuse = False
            trainable_variables = []
    for i in xrange(num_gpus):
        with tf.device("/gpu:%d" % i):
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
                        decoder_mask[start:end, 1:target_max_seq_length])
                    losses_ML.append(loss_ML)
        
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

                    max_final_length = tf.reduce_max(final_lengths)
                    padded_sample_ids = tf.pad(
                        sample_ids, 
                        [[0, 0], 
                         [0, max_iterations - max_final_length]])
                    sample_ids_lst.append(
                        padded_sample_ids[:, :max_iterations]) # truncate length
                    final_lengths_lst.append(final_lengths)
                
                # Compute tower gradients
                grads = compute_grads(loss_ML, optimizer, trainable_variables)
                tower_grads.append(grads)
    
    with tf.device('/cpu:0'):
        with tf.variable_scope("seq2seq"):
            # Concat sample ids and their respective lengths
            batch_sample_ids = tf.concat(sample_ids_lst, axis=0)
            batch_final_lengths = tf.concat(final_lengths_lst, axis=0)

            avg_loss_ML = tf.reduce_mean(
                tf.stack(losses_ML, axis=0)) # Thus, the effective batch size is actually batch_size_per_gpu
    
            apply_gradients_op = apply_grads(optimizer, tower_grads)   
                
    return (batch_sample_ids, batch_final_lengths, 
            avg_loss_ML, apply_gradients_op)

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    with tf.device('/cpu:0'):
        # Create all placeholders
        (input_seqs, input_seq_lengths, 
         target_seqs, target_seq_lengths,
         is_training) = create_placeholders(batch_size)

    (batch_sample_ids, batch_final_lengths, 
     avg_loss_ML, apply_gradients_op) = build_seq2seq(
        input_seqs, target_seqs, 
        input_seq_lengths, target_seq_lengths, 
        is_training)

with graph.as_default():    
    init = tf.global_variables_initializer()
    saver = get_saver()

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

    target_lst = append(prepend(target_lst, start_token), end_token)
    
    if (mode == "train"):
        training_flag = True
    else:
        training_flag = False

    num_source = len(source_lst)
    num_batches = num_source // batch_size
    
    responses = []
    log_perplexities = []
    for i in xrange(num_batches):
        start = batch_size * i
        end = start + batch_size
        
        source = source_lst[start:end]
        source_lengths = [len(seq) for seq in source]
        padded_source = pad(source, source_lengths)
        
        # Initialize feed_dict
        feed_dict = {
            input_seqs: padded_source,
            input_seq_lengths: source_lengths,
            is_training: training_flag}       
        
        if mode == "train":            
            target = target_lst[start:end]
            target_lengths = [len(seq) for seq in target]
            
            feed_dict[target_seqs] = pad(target, target_lengths)
            feed_dict[target_seq_lengths] = target_lengths
            (log_perplexity, _) = sess.run(
                [avg_loss_ML, apply_gradients_op],
                feed_dict=feed_dict)
            log_perplexities.append(log_perplexity)
     
            print("Epoch %d batch %d perplexity %.1f" % (epoch, i, exp(log_perplexity)))
        else:            
            [ids, lengths] = sess.run(
                [batch_sample_ids, batch_final_lengths],
                feed_dict=feed_dict)
            
            batch_responses = [
                [index for index in response[:length]]
                for (response, length) 
                in zip(ids.tolist(), lengths.tolist())]
            responses.extend(batch_responses)

            print("Finished testing epoch %d batch %d" % (i, epoch))
    
    if mode == "train":
        print("Average perplexity of epoch %d is: %.1f" % (epoch, exp(avg(log_perplexities))))
        saver.save(sess, data_path + "LFT_%d" % epoch)

        print("Checkpoint saved for epoch %d." % epoch)
    else:
        return responses

num_epochs = total_epochs - start_epoch

config = gpu_config()    
with tf.Session(graph=graph, config=config) as sess:
    sess.run(init)
    print("Initialized.") 

    if force_restore or start_epoch > 0:
        if force_restore:
            restore_ckpt = force_restore_point
        else:
            restore_ckpt = data_path + "LFT_%d" % (start_epoch - 1)
        saver.restore(sess, restore_ckpt)
        print("Restored checkpoint %s." % restore_ckpt)

    for i in xrange(num_epochs):
        (labeled_source_train, labeled_target_train) = shuffle(
            labeled_source_train, labeled_target_train)
        
        # Train
        run_seq2seq(
            sess, labeled_source_train, labeled_target_train, 
            "train", i + start_epoch)

        # Test
        if ((i + start_epoch + 1 >= 10)
            and (i + start_epoch + 1) % 5 == 0):
            filename = data_path + "LFT_result_%d.txt" % (i + start_epoch)
            
            polite_responses = run_seq2seq(
                sess, source_test_polite, target_test, "test", i + start_epoch)
            neutral_responses = run_seq2seq(
                sess, source_test_neutral, target_test, "test", i + start_epoch)
            rude_responses = run_seq2seq(
                sess, source_test_rude, target_test, "test", i + start_epoch)

            assert len(polite_responses) == len(neutral_responses) == len(rude_responses)

            num_responses = len(polite_responses)
            zipped = zip_lsts(
                [source_test[:num_responses], 
                 target_test[:num_responses],
                 polite_responses, 
                 neutral_responses, 
                 rude_responses])

            flattened = [decode2string(index2token, sent, remove_END_TOKEN=True) 
                         for tp in zipped for sent in tp]

            # now we mark sentences that are generated by our model
            marked_G = [("G: " + sent) 
                        if k % 5 == 1 else sent
                        for (k, sent) in enumerate(flattened)]
            marked_P = [("P: " + sent) 
                        if k % 5 == 2 else sent
                        for (k, sent) in enumerate(marked_G)]
            marked_N = [("N: " + sent) 
                        if k % 5 == 3 else sent
                        for (k, sent) in enumerate(marked_P)]
            marked_R = [("R: " + sent) 
                        if k % 5 == 4 else sent
                        for (k, sent) in enumerate(marked_N)]

            write_lines(filename, marked_R)

