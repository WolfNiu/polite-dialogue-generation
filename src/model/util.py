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


# In[ ]:


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
    
dropout_rate = 0.2
attention_size = 512
attention_layer_size = 256

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


# In[ ]:


def concat_states(states):
    state_lst = []
    for state in states:
        state_lst.append(tf.contrib.rnn.LSTMStateTuple(state[0], state[1]))
    return tuple(state_lst)

def get_keep_prob(dropout_rate, is_training):
    keep_prob = tf.cond(
        is_training, 
        lambda: tf.constant(1.0 - dropout_rate),
        lambda: tf.constant(1.0))
    return keep_prob

def dropout(cell, keep_prob, input_size):
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, dtype=tf.float32)        
    return cell_dropout

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

def create_MultiRNNCell(hidden_sizes, keep_prob, num_proj=None, 
                        memory=None, memory_seq_lengths=None, 
                        reuse=False):
    """
    Only the last layer has projection and attention

    Args:
        hidden_sizes: a list of hidden sizes for each layer
        num_proj: the projection size
    Returns:
        A cell or a wrapped rnn cell
    """
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
    
def lstm(input_size, hidden_size, keep_prob, reuse):
    cell = tf.contrib.rnn.LSTMCell(
        hidden_size, use_peepholes=True, # allow implementation of LSTMP
        initializer=tf.contrib.layers.xavier_initializer(),
        forget_bias=1.0, reuse=reuse)
    cell_dropout = tf.contrib.rnn.DropoutWrapper(
        cell,
        output_keep_prob=keep_prob,
        variational_recurrent=True, input_size=input_size, dtype=tf.float32)
    return cell_dropout

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

def bidirecitonal_dynamic_lstm(cell_fw, cell_bw, inputs, seq_lengths):
    (outputs, final_states) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw, cell_bw, inputs,                    
        sequence_length=seq_lengths,
        dtype=tf.float32,
        swap_memory=True)
    outputs_concat = tf.concat(outputs, axis=2)

    (final_states_fw, final_states_bw) = final_states
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


def compute_grads(loss, optimizer, var_list=None):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    valid_grads = [
        (grad, var) 
        for (grad, var) in grads 
        if grad is not None]
    if len(valid_grads) != len(var_list):
        print("Warning: some grads are None.")
    return valid_grads

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

def get_bad_mask(seqs):
    bad_tensor = tf.convert_to_tensor(bad_indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(bad_tensor, [len(bad_indices), 1, 1]))
    bad_mask = tf.logical_not(
        tf.reduce_any(bool_matrix, axis=0))
    return bad_mask

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


def get_sequence_mask(seq_lengths, dtype=tf.bool):
    max_seq_length = tf.reduce_max(seq_lengths)
    sequence_mask = tf.sequence_mask(
        seq_lengths, maxlen=max_seq_length,
        dtype=dtype)
    return sequence_mask

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

def get_BLEUs(refs, ref_lengths, hyps, hyp_lenghts):
    BLEUs = tf.py_func(
        calculate_BLEUs,
        [refs, ref_lengths, hyps, hyp_lenghts],
        tf.float32, stateful=False)
    return BLEUs

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

def tile_multi_cell_state(states):
    return tuple([tile_single_cell_state(state) for state in states])

"""
Speicify configurations of GPU
"""
def gpu_config():
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'    
    return config

def get_saver(var_scope=None):
    saver = tf.train.Saver(
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope))
    return saver

def pad_and_truncate(sample_ids, lengths, max_iterations):
    max_length = tf.reduce_max(lengths)
    padded_sample_ids = tf.pad(
        sample_ids,
        [[0, 0],
         [0, max_iterations - max_length]])
    truncated_sample_ids = padded_sample_ids[:, :max_iterations] # truncate length
    return truncated_sample_ids

def get_mask(seqs, indices):
    tensor = tf.convert_to_tensor(indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(tensor, [len(indices), 1, 1]))
    mask = tf.reduce_any(bool_matrix, axis=0)
    return mask

