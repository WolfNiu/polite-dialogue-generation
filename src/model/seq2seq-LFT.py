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
from src.basic.util import (shuffle, remove_duplicates, 
                            unzip_lst, zip_lsts,
                            prepend, append,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)

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
# num_gpus = 8
num_gpus = 1
assert batch_size % num_gpus == 0
batch_size_per_gpu = batch_size // num_gpus


"""
Load pickled lists
"""

data_path = "/usr/xtmp/tn9td/vocab/preprocessing/"

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


# In[9]:


def zip_remove_duplicates_unzip(lsts):
    zipped = zip_lsts(lsts)
    zipped_without_duplicates = remove_duplicates(zipped)
    unzipped = unzip_lst(zipped_without_duplicates)
    return unzipped


# In[10]:


[source_train, target_train] = zip_remove_duplicates_unzip([source_train, target_train])
# [source_test, target_test] = zip_remove_duplicates_unzip([source_test, target_test])


# In[11]:


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


# In[12]:


tags = ["<person>", "<number>", "<continued_utterance>"]

ner_tokens = [token2index[token] for token in tags]
unk_indices = [unk_token, ner_tokens[2]]


# In[13]:


bad_words = read_lines(
    "/usr/project/xtmp/tn9td/vocab/swear_words_wikitionary.txt")
bad_indices = [token2index[word] 
               for word in bad_words 
               if word in vocab]
good_words = read_lines(
    "/usr/project/xtmp/tn9td/vocab/polite_words.txt")
good_indices = [token2index[word] 
               for word in good_words 
               if word in vocab]


# In[14]:


def split_triple_lst(triple_lst, indices):
    popped_indices = [i for (i, triple) 
                      in enumerate(triple_lst) 
                      if len(set(triple[1]).intersection(set(indices))) > 0]
    kept_lst = [triple_lst[i] for (i, triple) 
                in enumerate(triple_lst)
                if i not in popped_indices]
    popped_lst = [triple_lst[i] for i in popped_indices]
    assert len(kept_lst) + len(popped_lst) == len(triple_lst)
    return (kept_lst, popped_lst)


# In[15]:


print([len(triple_lst) for triple_lst in triple_lsts])


# In[16]:


"""
Before reorganizing: [40390, 176317, 64253]
After reorganizing: [80287, 145321, 55352]
"""

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


# In[17]:


if reorganize:
    # Store reorganized utterances
    for (filename, triple_lst) in zip(filenames[13:], triple_lsts):
        dump_pickle(data_path + filename, triple_lst)


# In[18]:


len(load_pickle("/usr/xtmp/tn9td/vocab/preprocessing/polite_movie_target.pkl"))


# In[ ]:


# """
# Randomly sample "num_polite" number of neutral examples
# """
# num_polite = len(triple_lsts[0])
# np.random.shuffle(triple_lsts[1])
# triple_lsts[1] = triple_lsts[1][:num_polite]


# In[ ]:


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


# In[1]:


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


# In[ ]:


def concat_states(states):
    state_lst = []
    for state in states:
        state_lst.append(tf.contrib.rnn.LSTMStateTuple(state[0], state[1]))
    return tuple(state_lst)


# In[ ]:


def get_keep_prob(dropout_rate, is_training):
    keep_prob = tf.cond(
        is_training, 
        lambda: tf.constant(1.0 - dropout_rate),
        lambda: tf.constant(1.0))
    return keep_prob


# In[ ]:


def dropout(cell, keep_prob, input_size):
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, dtype=tf.float32)        
    return cell_dropout


# In[ ]:


def create_cell(input_size, hidden_size, keep_prob, num_proj=None, 
                memory=None, memory_seq_lengths=None, reuse=False):
    cell = tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, # peephole: allow implementation of LSTMP
        initializer=tf.contrib.layers.xavier_initializer(),
#         num_proj=num_proj, # testing, testing...
        forget_bias=1.0, reuse=reuse)
    cell = dropout(cell, keep_prob, input_size)
    # Note that the attention wrapper HAS TO come before projection wrapper,
    # Otherwise the attention weights will not work correctly.
    if memory is not None:
        cell = attention(cell, memory, memory_seq_lengths)
    if num_proj is not None:
        cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_proj)
    return cell


# In[ ]:


"""
Only the last layer has projection and attention

Args:
    hidden_sizes: a list of hidden sizes for each layer
    num_proj: the projection size
Returns:
    A cell or a wrapped rnn cell
"""
def create_MultiRNNCell(hidden_sizes, keep_prob, num_proj=None, 
                        memory=None, memory_seq_lengths=None, 
                        reuse=False):
    assert len(hidden_sizes) > 0
    
    if len(hidden_sizes) == 1:
        cell_first = create_cell(
            (embedding_size + attention_size), hidden_sizes[0], keep_prob, 
            num_proj=num_proj, 
            memory=memory, memory_seq_lengths=memory_seq_lengths, 
            reuse=reuse)
        return cell_first
    else: # if there are at least two layers        
        cell_first = create_cell(
            embedding_size, hidden_sizes[0], 
            keep_prob, 
            num_proj=None, 
            memory=None, memory_seq_lengths=None, 
            reuse=reuse)
        cell_last = create_cell(
            hidden_sizes[-2], hidden_sizes[-1],
            keep_prob,
            num_proj=num_proj,
            memory=memory, memory_seq_lengths=memory_seq_lengths, 
            reuse=reuse)
        cells_in_between = [
            create_cell(
                previous_hidden_size, hidden_size, 
                keep_prob, 
                num_proj=None, 
                memory=None, memory_seq_lengths=None, 
                reuse=reuse)
            for (previous_hidden_size, hidden_size)
            in zip(hidden_sizes[0:(-2)], hidden_sizes[1:(-1)])]
        return tf.contrib.rnn.MultiRNNCell(
            [cell_first] + cells_in_between + [cell_last])


# In[ ]:


def lstm(input_size, hidden_size, keep_prob, reuse):
    cell = tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, # allow implementation of LSTMP
        initializer=tf.contrib.layers.xavier_initializer(),
        forget_bias=1.0, reuse=reuse)
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
#         input_keep_prob=keep_prob, 
        output_keep_prob=keep_prob,
        variational_recurrent=True, input_size=input_size, dtype=tf.float32)
    return cell_dropout


# In[ ]:


def create_placeholders(batch_size):
    input_seqs = tf.placeholder(
        tf.int32, shape=[batch_size, None], name="input_seqs")
    input_seq_lengths = tf.placeholder(
        tf.int32, shape=[batch_size], name="input_seq_lengths")
    target_seqs = tf.placeholder(
        tf.int32, shape=[batch_size, None], name="target_seqs")
    target_seq_lengths = tf.placeholder(
        tf.int32, shape=[batch_size], name="target_seq_lengths")
    
    is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
    
    return (input_seqs, input_seq_lengths, 
            target_seqs, target_seq_lengths,
            is_training)


# In[ ]:


def create_embedding(embedding_word2vec_politeness, embedding_word2vec_movie,
                     shared_vocab_size_politeness, shared_vocab_size_movie,
                     new_vocab_size_politeness, new_vocab_size_movie, 
                     model):
    embedding_unk = tf.get_variable(
        "embedding_unk_%s" % model,
        shape=[1, embedding_size], 
        initializer=tf.contrib.layers.xavier_initializer())
    embedding_politeness_original = tf.get_variable(
        "embedding_politeness_original_%s" % model,
        shape=[shared_vocab_size_politeness, embedding_size],
        initializer=tf.constant_initializer(embedding_word2vec_politeness),
        trainable=True) # change to false for experiments
    embedding_politeness_new = tf.get_variable(
        "embedding_politeness_new_%s" % model,
        shape=[new_vocab_size_politeness, embedding_size], 
        initializer=tf.contrib.layers.xavier_initializer())
    embedding_movie_original = tf.get_variable(
        "embedding_movie_original_%s" % model,
        shape=[shared_vocab_size_movie, embedding_size],
        initializer=tf.constant_initializer(embedding_word2vec_movie),
        trainable=True)
    embedding_movie_new = tf.get_variable(
        "embedding_movie_new_%s" % model,
        shape=[new_vocab_size_movie, embedding_size], 
        initializer=tf.contrib.layers.xavier_initializer())
    
    # Have to do it in this order, otherwise UNK token won't be 0
    embedding = tf.concat(
        [embedding_unk, 
         embedding_politeness_original, embedding_politeness_new,
         embedding_movie_original, embedding_movie_new],
        axis=0)
    
    return embedding


# In[ ]:


def dynamic_lstm(cell, inputs, seq_lengths, initial_state, reuse=False):
    (outputs, final_state) = tf.nn.dynamic_rnn(
        cell,
        inputs,
        sequence_length=seq_lengths,
        initial_state=initial_state,
        dtype=tf.float32,
        swap_memory=True,
        time_major=False)
    return (outputs, final_state)


# In[ ]:


def bidirecitonal_dynamic_lstm(cell_fw, cell_bw, inputs, seq_lengths):
    (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs,                    
        sequence_length=seq_lengths,
        dtype=tf.float32,
        swap_memory=True)
    outputs_concat = tf.concat(outputs, axis=2)

    (final_states_fw, final_states_bw) = final_states
    if num_layers_decoder == 1:
        final_state_fw_c = final_states_fw[0]
        final_state_fw_h = final_states_fw[1]
        final_state_bw_c = final_states_bw[0]
        final_state_bw_h = final_states_bw[1]        
    else:
        final_states_concat = [
            tf.contrib.rnn.LSTMStateTuple(
                tf.concat(
                    [final_state_fw.c, final_state_bw.c], 
                    axis=1),
                tf.concat(
                    [final_state_fw.h, final_state_bw.h], 
                    axis=1))
            for (final_state_fw, final_state_bw)
            in zip(final_states_fw, final_states_bw)]
    return (outputs_concat, tuple(final_states_concat))


# In[ ]:


"""
Copied from: https://github.com/tensorflow/tensorflow/blob/r0.7/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py

Calculate the average gradient for each shared variable across all towers.
Note that this function provides a synchronization point across all towers.
Args:
    tower_grads: List of lists of (gradient, variable) tuples. 
        The outer list is over individual gradients. 
        The inner list is over the gradient calculation for each tower.
Returns:
    List of pairs of (gradient, variable) where the gradient has been 
    averaged across all towers.
"""
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
        # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

        # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, axis=0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1] # [0]: first tower [1]: ref to var
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# In[ ]:


"""
Compute average gradients, perform gradient clipping and apply gradients
Args:
    tower_grad: gradients collected from all GPUs
Returns:
    the op of apply_gradients
"""

def apply_grads(optimizer, tower_grads):
    # averaging over all gradients
    avg_grads = average_gradients(tower_grads)

    # Perform gradient clipping
    (gradients, variables) = zip(*avg_grads)
    (clipped_gradients, _) = tf.clip_by_global_norm(gradients, clipping_threshold)

    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = optimizer.apply_gradients(zip(clipped_gradients, variables))

    return apply_gradients_op


# In[ ]:


def apply_multiple_grads(optimizer, tower_grads_lst):
    if tower_grads_lst == []:
        print("Warning: empty tower grads list!")
    
    avg_grads_lst = [average_gradients(tower_grads) 
                     for tower_grads in tower_grads_lst]
    
    # First get the variables out (only need to be done once)
    (_, variables) = zip(*(avg_grads_lst[0]))
    
    # Sum corresponding gradients
    gradients_lst = []
    for avg_grads in avg_grads_lst:
        (gradients, _) = zip(*avg_grads)
        gradients_lst.append(gradients)
        
    zipped_gradients_lst = zip_lsts(gradients_lst)
    summed_gradients = [tf.add_n(zipped_gradients) 
                        for zipped_gradients 
                        in zipped_gradients_lst]
    
    # Perform gradient clipping
    (clipped_gradients, _) = tf.clip_by_global_norm(summed_gradients, clipping_threshold)
    
    # Apply the gradients to adjust the shared variables.
    apply_gradients_op = optimizer.apply_gradients(zip(clipped_gradients, variables))    
    
    return apply_gradients_op


# In[ ]:


def compute_grads(loss, optimizer, var_list=None):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    valid_grads = [
        (grad, var) 
        for (grad, var) in grads 
        if grad is not None]
    if len(valid_grads) != len(var_list):
        print("Warning: some grads are None.")
    return valid_grads


# In[ ]:


def attention(cell, memory, memory_seq_lengths):
    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        attention_size, 
        memory,
        memory_sequence_length=memory_seq_lengths)
    cell_attention = tf.contrib.seq2seq.AttentionWrapper(
        cell, attention_mechanism,
        attention_layer_size=256, # so that context and LSTM output is mixed together
        alignment_history=False, # Set to "False" for beam search!!
        output_attention=False)# behavior of BahdanauAttention
    return cell_attention


# In[ ]:


def decode(cell, helper, initial_state):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state)
    (decoder_outputs, _, final_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True,
        maximum_iterations=max_iterations, swap_memory=True)
    return (decoder_outputs, final_lengths)


# In[ ]:


def get_bad_mask(seqs):
    bad_tensor = tf.convert_to_tensor(bad_indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(bad_tensor, [len(bad_indices), 1, 1]))
    bad_mask = tf.logical_not(
        tf.reduce_any(bool_matrix, axis=0))
    return bad_mask


# In[ ]:


"""
returns:
    A mask that has False where 'unk' is present
"""
def get_unk_mask(seqs):
#     unk_mask = tf.logical_not(
#         tf.equal(seqs, unk_token))
    unk_tensor = tf.convert_to_tensor(unk_indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(unk_tensor, [len(unk_indices), 1, 1]))
    unk_mask = tf.logical_not(
        tf.reduce_any(bool_matrix, axis=0))
    return unk_mask


# In[ ]:


def get_sequence_mask(seq_lengths, dtype=tf.bool):
    max_seq_length = tf.reduce_max(seq_lengths)
    sequence_mask = tf.sequence_mask(
        seq_lengths, maxlen=max_seq_length,
        dtype=dtype)
    return sequence_mask


# In[30]:


"""
Calculate single reference BLEU 
with both input and output being numpy arrays
Args:
    n: consider up to n-grams
"""
def calculate_BLEU(reference, hypothesis, n=2):
    # Remove punctuations and special tokens(if any) from both ref and hyp
    [ref, hyp] = [
        [token for token in sent.tolist() 
         if token not in (punctuations + special_tokens)]
        for sent in [reference, hypothesis]]
    # Convert indices to strings
    [str_reference, str_hypothesis] = [
        list(map(str, sent))
        for sent
        in [ref, hyp]]
    
    weights = [1 / n] * n
    score = np.asarray(
        sentence_bleu([str_reference], # the [] for 'str_reference' is required 
                      str_hypothesis, weights=weights,
                      smoothing_function=SmoothingFunction().method0),
        dtype=np.float32)
    return score


# In[31]:


def calculate_BLEUs(reference, reference_lengths,
                    hypothesis, hypothesis_lengths):
    ref_lengths = reference_lengths.tolist()
    hyp_lengths = hypothesis_lengths.tolist()
    
    scores = [
        calculate_BLEU(reference[i, :ref_length], hypothesis[i, :hyp_length]) 
        for (i, (ref_length, hyp_length))
        in enumerate(zip(ref_lengths, hyp_lengths))]
    stacked_scores = np.stack(scores, axis=0)
    return stacked_scores


# In[ ]:


def get_BLEUs(refs, ref_lengths, hyps, hyp_lenghts):
    BLEUs = tf.py_func(
        calculate_BLEUs,
        [refs, ref_lengths, hyps, hyp_lenghts],
        tf.float32, stateful=False)
    return BLEUs


# In[2]:


def get_valid_mask(inputs):
    valid_mask = tf.cast(
        tf.logical_and(
            tf.not_equal(inputs, 0),
            tf.not_equal(inputs, end_token)),
        tf.float32)
    return valid_mask


# In[ ]:


def pad_tensor(tensor, lengths):
    max_length = tf.reduce_max(lengths)
    padded = tf.pad(
        tensor, 
        [[0, 0], 
         [0, max_iterations - max_length]])
    return padded


# In[ ]:


"""
Takes in a cell state and return its tiled version
"""
def tile_single_cell_state(state):
    if isinstance(state, tf.Tensor):
        s = tf.contrib.seq2seq.tile_batch(state, beam_width)
        if s is None:
            print("Got it!")
            print(state)
        return s
    elif isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        return tf.contrib.rnn.LSTMStateTuple(
            tile_single_cell_state(state.c), tile_single_cell_state(state.h))
    elif isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
        return tf.contrib.seq2seq.AttentionWrapperState(
            tile_single_cell_state(state.cell_state), tile_single_cell_state(state.attention), 
            state.time, tile_single_cell_state(state.alignments), 
            state.alignment_history)
    return None


# In[ ]:


def tile_multi_cell_state(states):
    return tuple([tile_single_cell_state(state) for state in states])


# In[ ]:


def get_saver(var_scope=None):
    saver = tf.train.Saver(
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope))
    return saver


# In[ ]:


"""
To train with both politeness score and BLEU, 
we need to run the decoder three times for each example during training
1. feed each step with ground truth --> loss_ML
2. sampling for each step for computing loss_polite and loss_BLEU
3. argmax each step to get baseline for BLEU RL training

Classifier gives:
1. politeness score for each sampled sentence
2. (optional) first derivative saliency
"""


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


# In[ ]:


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


# In[ ]:


with graph.as_default():    
    init = tf.global_variables_initializer()
    saver = get_saver()


# In[ ]:


"""
Speicify configurations of GPU
"""
def gpu_config():
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'    
    return config


# In[ ]:


def avg(lst):
    avg = sum(lst) / len(lst)
    return avg


# In[ ]:


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


# In[ ]:


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


# In[ ]:


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

#     polite_responses = run_seq2seq(
#         sess, source_test_polite, target_test, "test", start_epoch - 1)  

#     dump_pickle(data_path + "LFT_%d_infer_indexed.pkl" % (start_epoch - 1), polite_responses)
        
#     polite_responses = run_seq2seq(
#         sess, source_test_polite, target_test, "test", start_epoch - 1)
    
#     dump_pickle(
#         "/usr/project/xtmp/tn9td/vocab/LFT_%d_infer.pkl" % (start_epoch - 1),
#         polite_responses)
    
    for i in xrange(num_epochs):
#         # First test if the checkpoint already exists
#         ckpt = "%sLFT_%d" % (data_path, i + start_epoch)
#         fp = Path(ckpt)
#         if fp.is_file():
#             saver.restore(sess, ckpt)
#             continue
    
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

