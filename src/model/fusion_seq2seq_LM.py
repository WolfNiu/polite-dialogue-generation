# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import tensorflow as tf
import numpy as np
import os
from math import exp
import itertools

import sys
sys.path.append("../..")
from src.basic.util import (shuffle, remove_duplicates, 
                            zip_lsts, unzip_lst,
                            prepend, append,
                            load_pickle, load_pickles, 
                            dump_pickle, dump_pickles, 
                            build_dict, read_lines, write_lines, 
                            group_lst, decode2string)

def parse_args():
    parser.add_argument(
        "--ckpt_generator", type=str, default="../checkpoint/fusion/",
        help="path to model files")
    parser.add_argument(
        "--ckpt_classifier", type=str, default="../checkpoint/classifier/politeness_classifier_3")
    parser.add_argument(
        "--test", action="store_true", help="whether we are testing, default to False")
    args = parser.parse_args()
    return args


def zip_remove_duplicates_unzip(lsts):
    zipped = zip_lsts(lsts)
    zipped_without_duplicates = remove_duplicates(zipped)
    unzipped = unzip_lst(zipped_without_duplicates)
    return unzipped

args = parse_args()

"""
Load pickled lists
"""
current_dir = os.getcwd()

data_path = "../data/"
restore_path = args.ckpt_generator

filenames = [
    "vocab_all.pkl",
    "shared_vocab_politeness.pkl", "shared_vocab_movie.pkl",
    "new_vocab_politeness.pkl", "new_vocab_movie.pkl",
    "movie_train_source.pkl", "movie_train_target.pkl",
    "movie_valid_source.pkl", "movie_valid_target.pkl",
    "movie_test_source.pkl", "movie_test_target.pkl",
    "embedding_word2vec_politeness.pkl", "embedding_word2vec_movie.pkl"]

files = [os.path.join(current_dir, data_path, filename) for filename in filenames]

data = load_pickles(files)

vocab = data[0]
shared_vocab_politeness = data[1]
shared_vocab_movie = data[2]
new_vocab_politeness = data[3]
new_vocab_movie = data[4]
shared_vocab_size_politeness = len(shared_vocab_politeness)
shared_vocab_size_movie = len(shared_vocab_movie)
source_train = data[5] + data[7]
target_train = data[6] + data[8]
[source_train, target_train] = zip_remove_duplicates_unzip([source_train, target_train])
assert len(source_train) == len(target_train)
source_test = data[9]
target_test = data[10]
assert len(source_test) == len(target_test)
embedding_word2vec_politeness = data[11]
embedding_word2vec_movie = data[12]

# Load all the polite utterances
polite_lst = remove_duplicates(
    unzip_lst(
        load_pickle("../data/polite_movie_target.pkl"))[0])
print("Loaded %d polite examples!" % len(polite_lst))

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


[polite_lst, target_train] = [
    append(prepend(lst, start_token), end_token) 
    for lst in [polite_lst, target_train]]

"""
flags
"""
no_unk = True

"""
Shared hyperparameters
"""
batch_size = 96
num_gpus = 8
assert batch_size % num_gpus == 0 # make sure batch_size divides evenly
batch_size_per_gpu = batch_size // num_gpus
clipping_threshold = 5.0 # threshold for gradient clipping
embedding_size = 300

"""
seq2seq hyperparameters
"""
hidden_size_encoder = 256
hidden_size_decoder = 512
num_layers_encoder = 4
num_layers_decoder = 2
learning_rate_seq2seq = 0.001
max_iterations = 35 # should be computed by 95 percentile of all sequence lengths
dropout_rate_seq2seq = 0.2
attention_size = 512
attention_layer_size = 256
start_tokens = [start_token] * batch_size

"""
LM hyperparameters
"""
hidden_sizes_LM = [2048, 512]
learning_rate_LM = 0.2
dropout_rate_LM = 0.5
num_steps_LM = 20 # number for truncated BPTT


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

def average_gradients(tower_grads):
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

def create_placeholders():
    input_seqs = tf.placeholder(
        tf.int32, shape=[batch_size, None], name="input_seqs")
    tf.add_to_collection("placeholder", input_seqs)
    input_seq_lengths = tf.placeholder(
        tf.int32, shape=[batch_size], name="input_seq_lengths")
    tf.add_to_collection("placeholder", input_seq_lengths)
    target_seqs = tf.placeholder(
        tf.int32, shape=[batch_size, None], name="target_seqs")
    tf.add_to_collection("placeholder", target_seqs)
    target_seq_lengths = tf.placeholder(
        tf.int32, shape=[batch_size], name="target_seq_lengths")
    tf.add_to_collection("placeholder", target_seq_lengths)
    
    init_states_c_1 = tf.placeholder(
        tf.float32, shape=[batch_size, hidden_sizes_LM[0]], name="init_states_c_1")
    tf.add_to_collection("placeholder_LM", init_states_c_1)
    init_states_h_1 = tf.placeholder(
        tf.float32, shape=[batch_size, hidden_sizes_LM[0]], name="init_states_h_1")
    tf.add_to_collection("placeholder_LM", init_states_h_1)
    init_states_c_2 = tf.placeholder(
        tf.float32, shape=[batch_size, hidden_sizes_LM[1]], name="init_states_c_2")
    tf.add_to_collection("placeholder_LM", init_states_c_2)
    init_states_h_2 = tf.placeholder(
        tf.float32, shape=[batch_size, hidden_sizes_LM[1]], name="init_states_h_2")
    tf.add_to_collection("placeholder_LM", init_states_h_2)
    
    initial_states = ((init_states_c_1, init_states_h_1),
                      (init_states_c_2, init_states_h_2))
    
    is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
    tf.add_to_collection("placeholder", is_training)
    
    fusion_ratio = tf.placeholder(dtype=tf.float32, shape=[])
    tf.add_to_collection("placeholder", fusion_ratio)
    
    return (input_seqs, input_seq_lengths, 
            target_seqs, target_seq_lengths, 
            initial_states, is_training,
            fusion_ratio)

def get_bad_mask(seqs):
    bad_tensor = tf.convert_to_tensor(bad_indices)
    bool_matrix = tf.equal(
        tf.expand_dims(seqs, axis=0),
        tf.reshape(bad_tensor, [len(bad_indices), 1, 1]))
    bad_mask = tf.logical_not(
        tf.reduce_any(bool_matrix, axis=0))
    return bad_mask

"""
Args:
    model: a string that is either "LM" or "seq2seq"
"""

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

def decode(cell, helper, initial_state):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state)
    (decoder_outputs, _, final_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True,
        maximum_iterations=max_iterations, swap_memory=True)
    return (decoder_outputs, final_lengths)

"""
Recursive function to get the TensorShape of an input state
(as input to tf.while_loop's shape_invariants)
It is recusion. Don't try this at home!
Args:
    input_state: the input state
Returns:
    shape of the input state
"""

def get_state_shape(input_state):
    if isinstance(input_state, tf.contrib.seq2seq.AttentionWrapperState):
        state_shape = tf.contrib.seq2seq.AttentionWrapperState(
            *[get_state_shape(state) for state in input_state])
    elif isinstance(input_state, tf.contrib.rnn.LSTMStateTuple):
        state_shape = tf.contrib.rnn.LSTMStateTuple(
            *[get_state_shape(state) for state in input_state])
    elif isinstance(input_state, tf.TensorArray):
        state_shape = tf.TensorShape(None)
    elif isinstance(input_state, tf.Tensor):
        state_shape = input_state.get_shape()
    elif isinstance(input_state, tuple):
        state_shape = tuple([get_state_shape(e) for e in input_state])
    else:
        print(input_state)
        raise ValueError(
            ("\nInput state is none of the following:\n"
             "1. AttentionWrapperState\n"
             "2. LSTMStateTuple\n"
             "3. TensorArray\n"
             "4. Tensor"
             "5. tuple\n"))
    return state_shape

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

def compute_grads(loss, optimizer, var_list=None):
    grads = optimizer.compute_gradients(loss, var_list=var_list)
    valid_grads = [
        (grad, var) 
        for (grad, var) in grads 
        if grad is not None]
    if len(valid_grads) != len(var_list):
        print("Warning: some grads are None.")
    return valid_grads

def get_sequence_mask(seq_lengths, dtype=tf.bool):
    max_seq_length = tf.reduce_max(seq_lengths)
    sequence_mask = tf.sequence_mask(
        seq_lengths, maxlen=max_seq_length,
        dtype=dtype)
    return sequence_mask

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


# In[ ]:


def build_LM(input_seqs, seq_lengths,
             initial_states, is_training):
    with tf.variable_scope("LM"):
        with tf.device('/cpu:0'):
            # Obtain sequence mask
            unk_mask = get_unk_mask(input_seqs)
            bad_mask = get_bad_mask(input_seqs)
            sequence_mask = tf.cast(
                tf.logical_and(
                    get_sequence_mask(seq_lengths), 
                    tf.logical_and(bad_mask, unk_mask)),
                tf.float32)

            embedding = create_embedding(
                embedding_word2vec_politeness, embedding_word2vec_movie,
                shared_vocab_size_politeness, shared_vocab_size_movie,
                new_vocab_size_politeness, new_vocab_size_movie, 
                "LM")            
            embedded_input_seqs = tf.nn.embedding_lookup(
                embedding, input_seqs)   
            keep_prob = get_keep_prob(dropout_rate_LM, is_training)

            optimizer = tf.train.AdagradOptimizer(
                learning_rate_LM, name="optimizer")

            final_states_c_1 = []
            final_states_h_1 = []
            final_states_c_2 = []
            final_states_h_2 = []
            tower_grads = [] # keep track of the gradients across all towers
            losses = []
            reuse = False
            trainable_variables = []
        for i in xrange(num_gpus):
            with tf.device("/gpu:%d" % i):
                if (i == 1):
                    reuse = True
                start = i * batch_size_per_gpu
                end = start + batch_size_per_gpu
                
                cell = create_MultiRNNCell(
                    hidden_sizes_LM, keep_prob,
                    num_proj=vocab_size, memory=None,
                    reuse=reuse)
                
                max_seq_length = tf.reduce_max(seq_lengths[start:end])
                
                initial_state = concat_states(
                    ((initial_states[0][0][start:end, :], 
                      initial_states[0][1][start:end, :]),
                     (initial_states[1][0][start:end, :], 
                      initial_states[1][1][start:end, :])))
                
                with tf.variable_scope("lstm", reuse=reuse):                    
                    (logits, final_state) = dynamic_lstm(
                        cell, 
                        embedded_input_seqs[start:end, :(max_seq_length - 1), :], # get rid of end_token
                        seq_lengths[start:end] - 1,
                        initial_state=initial_state,
                        reuse=reuse)
    
                final_states_c_1.append(final_state[0][0])
                final_states_h_1.append(final_state[0][1])
                final_states_c_2.append(final_state[1][0])
                final_states_h_2.append(final_state[1][1])
                
                loss = tf.contrib.seq2seq.sequence_loss(
                    logits, 
                    input_seqs[start:end, 1:max_seq_length], # get rid of start_token
                    sequence_mask[start:end, 1:max_seq_length])
                losses.append(loss)

                # Get trainable_variables 
                # (up to now we already have all the seq2seq trainable vars)
                if trainable_variables == []:
                    trainable_variables = tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES, 
                        scope="LM")
                
                grads = compute_grads(loss, optimizer, var_list=trainable_variables)
                tower_grads.append(grads)
                
        with tf.device('/cpu:0'):        
            # concatenate all final states
            final_state_c_1_concat = tf.concat(
                final_states_c_1, axis=0)
            final_state_h_1_concat = tf.concat(
                final_states_h_1, axis=0)
            final_state_c_2_concat = tf.concat(
                final_states_c_2, axis=0)
            final_state_h_2_concat = tf.concat(
                final_states_h_2, axis=0)
            
            # averaging over all losses        
            avg_loss = tf.reduce_mean(tf.stack(losses))

            # Apply the gradients to adjust the shared variables.
            apply_gradients_op = apply_grads(optimizer, tower_grads)
            
    return (cell, avg_loss, apply_gradients_op, 
            final_state_c_1_concat, final_state_h_1_concat, 
            final_state_c_2_concat, final_state_h_2_concat,
            embedding)


# In[ ]:


def build_seq2seq(input_seqs, target_seqs,
                  input_seq_lengths, target_seq_lengths, 
                  is_training):
    with tf.variable_scope("seq2seq"):
        with tf.device('/cpu:0'):
            reuse = False
            keep_prob = get_keep_prob(dropout_rate_seq2seq, is_training)
            
            # mask for decoder
            sequence_mask = get_sequence_mask(target_seq_lengths)
            
            if no_unk:
                unk_mask = tf.equal(target_seqs, unk_token)
                decoder_mask = tf.cast(
                    tf.logical_and(sequence_mask, tf.logical_not(unk_mask)),
                    tf.float32)
            else:
                decoder_mask = tf.cast(sequence_mask, tf.float32)
            
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
            optimizer = tf.train.AdamOptimizer(learning_rate_seq2seq)

            losses = []
            tower_grads = []
            sample_ids_lst = []
            final_lengths_lst = []
            reuse = False
            trainable_variables = []
            decoder_initial_state_lst = []
        for i in xrange(num_gpus):
            with tf.device('/gpu:%d' % i):
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
                    (encoder_outputs, encoder_final_state) = bidirecitonal_dynamic_lstm(
                        cell_fw, cell_bw, 
                        embedded_input_seqs[start:end, :input_max_seq_length, :],
                        input_seq_lengths[start:end])
                
                with tf.variable_scope("decoder", reuse=reuse):
                    decoder_cell = create_MultiRNNCell(
                        [hidden_size_decoder] * (num_layers_decoder),
                        keep_prob, num_proj=vocab_size,
                        memory=encoder_outputs,
                        memory_seq_lengths=input_seq_lengths[start:end],
                        reuse=reuse)

                    decoder_zero_state = decoder_cell.zero_state(
                        batch_size_per_gpu, tf.float32)
                    
                    state_last = decoder_zero_state[-1].clone(
                        cell_state=encoder_final_state[-1])
                    state_previous = encoder_final_state[:-1]
                    decoder_initial_state = state_previous + (state_last, ) # concat tuples
                    
                    decoder_initial_state_lst.append(decoder_initial_state)
                    
                    # Train branch
                    helper_train = tf.contrib.seq2seq.TrainingHelper(
                        embedded_target_seqs[
                            start:end, :target_max_seq_length - 1, :], # get rid of end_token
                        target_seq_lengths[start:end] - 1) # the length is thus decreased by 1

                    (decoder_outputs_train, _) = decode(
                        decoder_cell, helper_train,
                        decoder_initial_state)
                    (logits, _) = decoder_outputs_train
 
                    # Get trainable_variables 
                    # (up to now we already have all the seq2seq trainable vars)
                    if trainable_variables == []:
                        trainable_variables = tf.get_collection(
                            tf.GraphKeys.TRAINABLE_VARIABLES, 
                            scope="seq2seq")

                    loss = tf.contrib.seq2seq.sequence_loss(
                        logits,
                        target_seqs[start:end, 1:target_max_seq_length], # get rid of start_token
                        decoder_mask[start:end, 1:target_max_seq_length])
                    losses.append(loss)
                    grads = compute_grads(loss, optimizer, trainable_variables)
                    tower_grads.append(grads)
                                        
                    # Infer branch                    
                    helper_infer = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                        embedding, start_tokens[start:end], end_token)
                    (decoder_outputs_infer, final_lengths) = decode(
                        decoder_cell, helper_infer, decoder_initial_state)
                    (_, sample_ids) = decoder_outputs_infer
                    sample_ids_lst.append(
                        tf.pad(
                            sample_ids, 
                            [[0, 0], 
                             [0, max_iterations - tf.reduce_max(final_lengths)]]))
                    final_lengths_lst.append(final_lengths)
                    
    
        with tf.device('/cpu:0'):        
            # Compute average tower grads
            avg_loss = tf.reduce_mean(losses)
            apply_gradients_op = apply_grads(optimizer, tower_grads)

            batch_sample_ids = tf.concat(sample_ids_lst, axis=0)
            batch_final_lengths = tf.concat(final_lengths_lst, axis=0)
        
        
    return (decoder_cell, decoder_initial_state_lst, 
            batch_sample_ids, batch_final_lengths, 
            avg_loss, apply_gradients_op,
            embedding)                


# In[ ]:


def get_saver(var_scope=None):
    saver = tf.train.Saver(
        var_list=tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=var_scope))
    return saver


# In[ ]:


def cond(time, inputs, state_LM, state_decoder, finished, lengths, acc_sample_ids):
    return tf.logical_and(
        tf.logical_not(tf.reduce_all(finished)), # if any response is not finished
        tf.less(time, tf.convert_to_tensor(max_iterations)))


# In[ ]:


def body(time, inputs, state_LM, state_decoder, finished, lengths, acc_sample_ids):
    # Embedding
    embedded_inputs_LM = tf.nn.embedding_lookup(embedding_LM, inputs)
    embedded_inputs_seq2seq = tf.nn.embedding_lookup(embedding_seq2seq, inputs)
    # Run one step
    (logits_LM, next_state_LM) = cell_LM(
        embedded_inputs_LM, state_LM)
    (logits_seq2seq, next_state_decoder) = decoder_cell(
        embedded_inputs_seq2seq, state_decoder)
    # Apply fusion
    logits_fusion = (1 - fusion_ratio) * logits_seq2seq + fusion_ratio * logits_LM
    # Get sample ids and update accumulator
    sample_ids = tf.cast(tf.argmax(logits_fusion, axis=1), tf.int32)
    next_acc_sample_ids = acc_sample_ids.write(time, sample_ids) # time serves as index
    # update lengths
    lenghts_to_add = tf.cast(tf.logical_not(finished), tf.int32)
    next_lengths = lengths + lenghts_to_add
    # update finished vector
    next_finished = tf.logical_or(
        finished,
        tf.equal(sample_ids, end_token))
    # update step counter
    next_time = time + 1        
    return (next_time, sample_ids, next_state_LM, next_state_decoder, 
            next_finished, next_lengths, next_acc_sample_ids)


# In[ ]:


def get_while_loop_params(decoder_initial_state):
    # initial values
    init_time = tf.constant(0, shape=[])
    init_sample_ids = tf.constant(
        start_token, dtype=tf.int32, shape=[batch_size_per_gpu])
    init_finished = tf.constant(False, shape=[batch_size_per_gpu])
    init_state_LM = cell_LM.zero_state(batch_size_per_gpu, dtype=tf.float32)
    init_lengths = tf.zeros([batch_size_per_gpu], dtype=tf.int32)
    init_acc_sample_ids = tf.TensorArray(
        tf.int32, size=1, dynamic_size=True,
        clear_after_read=True, infer_shape=True, 
        element_shape=batch_size_per_gpu)
    initial_values = [
        init_time, init_sample_ids, init_state_LM, decoder_initial_state, 
        init_finished, init_lengths, init_acc_sample_ids]
    shape_invariants = [
        get_state_shape(initial_value) for initial_value in initial_values[:(-1)]]
    shape_invariants.append(tf.TensorShape(None)) # acc_sample_ids needs to be dynamic
#     shape_invariants = [tf.TensorShape(None)] * len(initial_values)
    return (initial_values, shape_invariants)


# In[ ]:


def pad_and_truncate(sample_ids, lengths):
    max_length = tf.reduce_max(lengths)
    padded_sample_ids = tf.pad(
         sample_ids, 
         [[0, 0], 
          [0, max_iterations - max_length]])
    truncated_sample_ids = padded_sample_ids[:, :max_iterations] # truncate length
    return truncated_sample_ids


# In[ ]:


def build_fusion(decoder_initial_state_lst):
    with tf.variable_scope("fusion"):
        sample_ids_lst = []
        final_lengths_lst = []
        for (i, decoder_initial_state) in enumerate(decoder_initial_state_lst):
            with tf.device("/gpu:%d" % i):
                (initial_values, shape_invariants) = get_while_loop_params(decoder_initial_state)

                loop_result = tf.while_loop(
                    cond, body, initial_values,
                    shape_invariants=shape_invariants,
                    swap_memory=True)
                
                sample_ids = tf.transpose(loop_result[-1].stack())
                final_lengths = loop_result[-2]
                
                sample_ids_lst.append(
                    pad_and_truncate(sample_ids, final_lengths))
                final_lengths_lst.append(final_lengths)
        
        batch_sample_ids = tf.concat(sample_ids_lst, axis=0)
        batch_final_lengths = tf.concat(final_lengths_lst, axis=0)
        
    return (batch_sample_ids, batch_final_lengths)


# In[ ]:


"""
Build a graph that can train a language model, 
a seq2seq model and fuse them to together when
doing inference.
"""
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    # Create placeholders for seq2seq & LM
    (input_seqs, input_seq_lengths, 
     target_seqs, target_seq_lengths,
     initial_states, is_training, 
     fusion_ratio) = create_placeholders()

    # Unpack initial_states for fetching in session
    ((init_states_c_1, init_states_h_1),
     (init_states_c_2, init_states_h_2)) = initial_states

    (decoder_cell, decoder_initial_state_lst,
     sample_ids_seq2seq, final_lengths_seq2seq, 
     loss_seq2seq, apply_gradients_op_seq2seq,
     embedding_seq2seq) = build_seq2seq(
        input_seqs, target_seqs, 
        input_seq_lengths, target_seq_lengths, 
        is_training)
        
    (cell_LM, loss_LM, apply_gradients_op_LM,
     final_state_c_1_concat, final_state_h_1_concat, 
     final_state_c_2_concat, final_state_h_2_concat,
     embedding_LM) = build_LM(
        input_seqs, input_seq_lengths,
        initial_states, is_training)
    
    (sample_ids_fusion, final_lengths_fusion) = build_fusion(
        decoder_initial_state_lst)


# In[ ]:


with graph.as_default():   
    init = tf.global_variables_initializer()
    
    saver = get_saver()
    saver_seq2seq = get_saver("seq2seq")
    saver_LM = get_saver("LM")


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


def run_fusion(sess, fusion_rate, source_lst):   
    training_flag = False

    num_batches = len(source_lst) // batch_size
        
    responses = []
    for i in xrange(num_batches):
        start = batch_size * i
        end = start + batch_size
        
        source = source_lst[start:end]
        source_lengths = [len(seq) for seq in source]
                
        feed_dict = {
            input_seqs: pad(source, source_lengths),
            input_seq_lengths: source_lengths,
            is_training: training_flag,
            fusion_ratio: fusion_rate}
        
        [ids, lengths] = sess.run(
            [sample_ids_fusion, final_lengths_fusion],
            feed_dict=feed_dict)
        batch_responses = [
            [index for index in response[:length]]
            for (response, length) 
            in zip(ids.tolist(), lengths.tolist())]
        responses.extend(batch_responses)
        print("Finished testing batch %d" % i)        
    return responses


# In[ ]:


def run_seq2seq(sess, source_lst, target_lst, mode, epoch):
    if (mode == "train"):
        training_flag = True
    else:
        training_flag = False

    num_batches = len(source_lst) // batch_size
        
    responses = []
    log_perplexities = []
    for i in xrange(num_batches):
        start = batch_size * i
        end = start + batch_size
        
        source = source_lst[start:end]
        source_lengths = [len(seq) for seq in source]
            
        feed_dict = {
            input_seqs: pad(source, source_lengths),
            input_seq_lengths: source_lengths,
            is_training: training_flag}
        
        if mode == "train":
            target = target_lst[start:end]
            target_lengths = [len(seq) for seq in target]
            
            feed_dict[target_seqs] = pad(target, target_lengths)
            feed_dict[target_seq_lengths] = target_lengths
            (log_perplexity, _) = sess.run(
                [loss_seq2seq, apply_gradients_op_seq2seq],
                feed_dict=feed_dict)
            log_perplexities.append(log_perplexity)
                        
            print("Batch %d perplexity %.1f" % (i, exp(log_perplexity)))
        else:
            [ids, lengths] = sess.run(
                [sample_ids_seq2seq, final_lengths_seq2seq],
                feed_dict=feed_dict)
            batch_responses = [
                [index for index in response[:length]]
                for (response, length) 
                in zip(ids.tolist(), lengths.tolist())]
            responses.extend(batch_responses)
            
            print("Finished testing batch %d" % i)
            
    if mode == "train":
        print("Average perplexity of batch %d is: %.1f" % (i, exp(avg(log_perplexities))))
        saver.save(sess, data_path + "seq2seq_%d" % epoch)
        print("Checkpoint saved for epoch %d." % epoch)
    else:
        num_responses = len(responses)
        source_lst = source_lst[:num_responses]
        target_lst = target_lst[:num_responses]
        source_and_responses = [
            [source, target, response] 
            for (source, target, response) 
            in zip(source_lst, target_lst, responses)]
        flattened_source_and_responses = [
            utterance 
            for group in source_and_responses 
            for utterance in group]

        text_lines = []
        for utterance in flattened_source_and_responses:
            tokens = [index2token[index] for index in utterance]
            text_lines.append((' ').join(tokens))
        
        print("Saving test result...")
        
        result_filename = data_path + "fusion_result_%d.txt" % epoch
        with open(result_filename, "w") as fp:
            for text_line in text_lines:
                print(text_line, file=fp)            


# In[2]:


def run_LM(all_sents, epoch, mode):
    
    if (mode == "train"):
        keep = 1.0 - dropout_rate_LM
    else:
        keep = 1.0
    
    global_index = batch_size # the next sentence to be learned
    assert(len(all_sents) >= batch_size)

    sents = all_sents[:batch_size]
    starts = [0] * batch_size
    ends = [num_steps_LM] * batch_size

    zero_state_1 = np.zeros([batch_size, hidden_sizes_LM[0]])
    zero_state_2 = np.zeros([batch_size, hidden_sizes_LM[1]])
    # Keep track of the initial states to be fed into the next batch
    state_c_1 = zero_state_1
    state_h_1 = zero_state_1
    state_c_2 = zero_state_2
    state_h_2 = zero_state_2
    
    average_losses = []
    batch_counter = 0
    while (True):
        sequence_lengths = [num_steps_LM] * batch_size # initialize sequence lengths
        inputs = []
        for (i, (sent, start, end)) in enumerate(zip(sents, starts, ends)): 
            if (global_index >= len(all_sents)):
                break
            
            if (start > len(sent) - 1): # if already finished the current sentence
                sents[i] = sent = all_sents[global_index]
                global_index += 1
                start = 0
                end = num_steps_LM
                # reset the initial states for the corresponding sentence
                state_c_1[i, :] = 0.0
                state_h_1[i, :] = 0.0
                state_c_2[i, :] = 0.0
                state_h_2[i, :] = 0.0
            if end > len(sent): # if one more batch until finishing the sentence
                sequence_lengths[i] = len(sent) - start
            inputs.append(sent[start:end]) # no need to worry about out of bounds
            starts[i] += num_steps_LM
            ends[i] += num_steps_LM
        
        if (global_index >= len(all_sents)):
            break
        
        feed_dict = {
            init_states_c_1: state_c_1,
            init_states_h_1: state_h_1,
            init_states_c_2: state_c_2,
            init_states_h_2: state_h_2,
            input_seqs: pad(inputs, sequence_lengths),
            input_seq_lengths: sequence_lengths}
        
        if mode == "train":
            feed_dict[is_training] = True
            (_, average_loss, state_c_1, state_h_1, state_c_2, state_h_2) = sess.run(
                [apply_gradients_op_LM, loss_LM,
                 final_state_c_1_concat, final_state_h_1_concat, 
                 final_state_c_2_concat, final_state_h_2_concat],
                feed_dict=feed_dict)
        else:
            feed_dict[is_training] = False
            (average_loss, state_c_1, state_h_1, state_c_2, state_h_2) = sess.run(
                [loss_LM,
                 final_state_c_1_concat, final_state_h_1_concat, 
                 final_state_c_2_concat, final_state_h_2_concat],
                feed_dict=feed_dict)
        average_losses.append(average_loss)
        print("Perplexity for batch %d: %.1f" % (batch_counter, exp(average_loss)))    
        batch_counter += 1
        
    if mode == "train":
        saver_LM.save(sess, data_path + "/LM_%d" % epoch)
        print("Checkpoint saved for epoch %d." % epoch)
    
    ppl = exp(sum(average_losses) / len(average_losses))
    print("Perplexity for the %s dataset: %.1f" % (mode, ppl))
    return ppl


# In[ ]:


def convert_list_to_str_list(lst):
    str_lst = [str(x) + ': ' for x in lst]
    return str_lst


# In[ ]:


fusion_rate_candidates = [0.33, 0.5]
num_epochs_seq2seq = 40
num_epochs_LM = 35

seq2seq_epoch = 38
LM_epoch = 21

config = gpu_config()    
with tf.Session(graph=graph, config=config) as sess:
    sess.run(init)
    print("Initialized.")
    
    ckpt_seq2seq = restore_path + "seq2seq_RL_%d" % seq2seq_epoch
    ckpt_LM = restore_path + "LM_%d" % LM_epoch
    
    saver_seq2seq.restore(sess, ckpt_seq2seq)
    print("Resotred checkpoint", ckpt_seq2seq)
    saver_LM.restore(sess, ckpt_LM)
    print("Resotred checkpoint", ckpt_LM)
       
    for i in xrange(num_epochs_seq2seq):
        run_seq2seq(sess, source_train, target_train, "train", i)
        (source_train, target_train) = shuffle(source_train, target_train)
        run_seq2seq(sess, source_test, target_test, "test", i)
    
    num_polite_examples = len(polite_lst)
    test_size = num_polite_examples // 10
    polite_lst_train = polite_lst[:(-test_size)]
    polite_lst_test = polite_lst[(-test_size):]
    
    min_ppl = 100000
    for i in xrange(num_epochs_LM):
        run_LM(polite_lst_train, i, "train")
        np.random.shuffle(polite_lst_train)
        ppl = run_LM(polite_lst_test, i, "test")
        if ppl < min_ppl:
            min_ppl = ppl
        else:
            print("Done with LM training. Stopped at epoch %d." % i)
            break
    
    # Run fusion model for different fusion rates    
    responses_lst = []
    for fusion_rate in fusion_rate_candidates:
        responses = run_fusion(sess, fusion_rate, source_test)
        
        dump_pickle(
            data_path + "/fusion_%.1f_%d_infer.pkl" % (fusion_rate, seq2seq_epoch), 
            responses)
        
        responses_lst.append(responses)

    num_responses = len(responses_lst[0])
    print("Generated %d responses for each fusion rate." % num_responses)
    
    # add in source sents and ground truths
    zipped_responses = zip_lsts(
        [source_test[:num_responses]] + 
        [target_test[:num_responses]] +
        responses_lst)
    
    # Write results to file
    filename = data_path + "fusion_responses_%.1f.txt" % fusion_rate_candidates[0]
    
    text_zipped_responses = [
        [label + decode2string(index2token, response, remove_END_TOKEN=True)
         for (label, response) in zip(
             ["", "G: "] + convert_list_to_str_list(fusion_rate_candidates),
             responses)]
        for responses in zipped_responses]
    
    flattened_text_responses = [
        response 
        for responses in text_zipped_responses
        for response in responses]
        
    write_lines(filename, flattened_text_responses)
        

