#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Todo list:
0. Use the Ubuntu data generator for glimpse training!
    Also do not feed the encoder final state (actually try both with different GPUs), 
    but keep track of last glimpse decoding final state 
    (keep track of it by a tf.Variable)
    reduce the learning rate to 0.0001
0.5. Initialize embedding with GLOVE/WIKI

1. Input and output vocab should be different! 
    • Input vocab should be the whole word2vec vocab, so that similar inputs can be embedded 
        to similar vectors (of course we need to set word2vec matrix's trainble=False). 
    • On the other hand, output vocab should be limited in order to avoid slow softmaxing.
    • When we are doing statistics on output vocab, 
        it should only contain words from target sentences!!
        Note: this point has nothing to do with OpenSubtitles
2. Think about how to obtain the basline
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange
import functools
import tensorflow as tf
from pprint import pprint
import sys
sys.path.append("../..")
from src.model.util import (attention, pad_and_truncate, dropout,
                            create_cell, create_MultiRNNCell, 
                            bidirecitonal_dynamic_lstm, get_mask,
                            average_gradients, compute_grads, apply_grads, 
                            tile_single_cell_state, tile_multi_cell_state)


# In[ ]:


def decode(cell, helper, initial_state, max_iterations, output_layer=None):
    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell, helper, initial_state, output_layer=output_layer)
    (decoder_outputs, _, final_lengths) = tf.contrib.seq2seq.dynamic_decode(
        decoder, impute_finished=True,
        maximum_iterations=max_iterations, swap_memory=True)
    return (decoder_outputs, final_lengths)


# In[ ]:


class Seq2Seq(object):
    def __init__(self,
                 batch_size,
                 shared_vocab_size_politeness, new_vocab_size_politeness,
                 shared_vocab_size_movie, new_vocab_size_movie,
                 embedding_word2vec_politeness, embedding_word2vec_movie,
                 embedding_size, hidden_size, num_layers,
                 max_iterations,
                 start_token, end_token, unk_indices,
                 attention_size=512, attention_layer_size=256,
                 beam_width=10, length_penalty_weight=1.0,
                 gpu_start_index=0, 
                 num_gpus=1, # set to 1 when testing
                 learning_rate=0.0001, 
                 clipping_threshold=5.0,
#                  str_input=False, vocab=None,
                 feed_final_state=True,
                 sampling_prob=0.25,
                 monotonic_attention=False,
                 proj_embedding=True,
                 continuous_label=False,
                 output_layer=False,
                 clip_loss=False, clip_ratio=0.2, decay_ratio=0.999):
        """
        Args:
            num_layers: set it to 2.
            max_iterations: should be set to max_iterations
        """
        self.batch_size = batch_size
        self.batch_size_per_gpu = batch_size // num_gpus
        self.shared_vocab_size_politeness = shared_vocab_size_politeness
        self.new_vocab_size_politeness = new_vocab_size_politeness
        self.shared_vocab_size_movie = shared_vocab_size_movie
        self.new_vocab_size_movie = new_vocab_size_movie
        self.vocab_size = (1 + # +1 for UNK_TOKEN
                           shared_vocab_size_politeness + new_vocab_size_politeness + 
                           shared_vocab_size_movie + new_vocab_size_movie)
        self.embedding_word2vec_politeness = embedding_word2vec_politeness
        self.embedding_word2vec_movie = embedding_word2vec_movie
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        assert self.hidden_size % 2 == 0
        self.num_layers = num_layers
        self.max_iterations = max_iterations
        self.start_token = start_token
        self.end_token = end_token
        self.unk_indices = unk_indices
        self.attention_size = attention_size
        self.attention_layer_size = attention_layer_size
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.clipping_threshold = clipping_threshold
#         self.str_input = str_input
#         self.vocab = vocab
        self.feed_final_state = feed_final_state
        self.sampling_prob = sampling_prob # probability for scheduled sampling
        self.monotonic_attention = monotonic_attention
        self.proj_embedding = proj_embedding
        self.continuous_label = continuous_label
        self.output_layer = True if output_layer else None
        self.clip_loss = clip_loss
        self.clip_ratio = clip_ratio
        self.decay_ratio = decay_ratio
        
        self.trainable_variables = []        
        with tf.variable_scope("seq2seq", reuse=tf.AUTO_REUSE):
            with tf.device('/cpu:0'):
                self.create_placeholders()
                self.avg_loss = tf.get_variable(
                    "avg_loss", shape=[],
                    initializer=tf.constant_initializer(4.0, dtype=tf.float32),
                    trainable=False)
                
#                 if self.str_input:
#                     source = self.indexed_source
#                     target = self.indexed_target
#                 else:
                source = self.source
                target = self.target
                
                # Note: Make sure batch size can be evenly divided by num_gpus
                split = functools.partial(tf.split, num_or_size_splits=num_gpus, axis=0)
                [source_lst, source_length_lst, target_lst, target_length_lst, start_tokens_lst] = [
                    split(tensor) for tensor 
                    in [source, self.source_length, target, self.target_length, self.start_tokens]]                
                
                optimizer = tf.train.AdamOptimizer(learning_rate)

                sample_ids_beam_lst = []
                final_lengths_beam_lst = []
                num_tokens_lst = []
                total_losses = []
                tower_grads = []
            for i in xrange(num_gpus):
                source_max_length = tf.reduce_max(source_length_lst[i])
                target_max_length = tf.reduce_max(target_length_lst[i])
                (num_tokens, total_loss, grads,
                 sample_ids_beam, final_lengths_beam) = self.one_iteration(
                    source_lst[i][:, :source_max_length], source_length_lst[i],
                    target_lst[i][:, :target_max_length], target_length_lst[i],
                    start_tokens_lst[i],
                    optimizer,
                    gpu_index=(gpu_start_index + i))
                sample_ids_beam_lst.append(
                    pad_and_truncate(sample_ids_beam, final_lengths_beam, 
                                     self.max_iterations))
                final_lengths_beam_lst.append(final_lengths_beam)
                num_tokens_lst.append(num_tokens)
                total_losses.append(total_loss)
                tower_grads.append(grads)

            with tf.device('/cpu:0'):
                self.batch_total_loss = tf.add_n(total_losses)
                self.batch_num_tokens = tf.add_n(num_tokens_lst)

                # Concat sample ids and their respective lengths
                self.batch_sample_ids_beam = tf.concat(
                    sample_ids_beam_lst, axis=0)
                self.batch_final_lengths_beam = tf.concat(
                    final_lengths_beam_lst, axis=0)

                self.apply_gradients_op = apply_grads(
                    optimizer, tower_grads)
    
    def one_iteration(self,
                      source, source_length,
                      target, target_length,
                      start_tokens,
                      optimizer,
                      gpu_index=0):
        with tf.device("/gpu:%d" % gpu_index):
            embedding = self.create_embedding()
            if self.output_layer:
                output_layer = self.create_output_layer()
            else:
                output_layer = None
            
            # Encoder
            (encoder_output, encoder_lengths, encoder_final_state) = self.encode(
                source, source_length, embedding)
            
            # Decoder cell & initial state
            (decoder_cell, decoder_initial_state) = self.get_decoder_cell_and_initial_state(
                encoder_output, encoder_lengths, encoder_final_state)
            
            (loss_ML, num_tokens, total_loss) = self.decode_train(
                tf.concat(
                    [tf.expand_dims(start_tokens, axis=1), target], axis=1),
                target_length + 1, 
                embedding,
                decoder_cell, decoder_initial_state, 
                output_layer=output_layer
            )
            
            # Get trainable variables
            # (up to now we already have all the seq2seq trainable vars)
            if self.trainable_variables == []:
                self.trainable_variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, scope="seq2seq")

            # Compute tower gradients
            grads = compute_grads(loss_ML, optimizer, self.trainable_variables)

            # Decoder -- beam (for inference)
            (sample_ids_beam, final_lengths_beam) = self.decode_beam(
                embedding, decoder_cell, decoder_initial_state, start_tokens, 
                output_layer=output_layer)
            
        return (num_tokens, total_loss, grads,
                sample_ids_beam, final_lengths_beam)
    
    def get_decoder_cell_and_initial_state(self, 
                                           encoder_outputs, encoder_lengths,
                                           encoder_final_state):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            num_proj = None if self.output_layer else self.vocab_size

            decoder_cell = create_MultiRNNCell(
                [self.hidden_size] * self.num_layers,
                self.keep_prob, 
                num_proj=num_proj,
                memory=encoder_outputs, 
                memory_seq_lengths=encoder_lengths,
                reuse=tf.AUTO_REUSE)

            decoder_zero_state = tf.cond(
                self.is_training,
                lambda: decoder_cell.zero_state(self.batch_size_per_gpu, tf.float32),
                lambda: decoder_cell.zero_state(
                    self.batch_size_per_gpu * self.beam_width, tf.float32))

            if self.feed_final_state:
                state_last = decoder_zero_state[-1].clone(
                    cell_state=encoder_final_state[-1])
                state_previous = encoder_final_state[:-1]
                decoder_initial_state = state_previous + (state_last, ) # concat tuples
            else:
                decoder_initial_state = decoder_zero_state
        return (decoder_cell, decoder_initial_state)

    def decode_beam(self, 
                    embedding, decoder_cell, 
                    decoder_initial_state, start_tokens,
                    output_layer=None):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            beam_search_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                decoder_cell, embedding, 
                start_tokens, self.end_token,
                decoder_initial_state, self.beam_width,
                output_layer=output_layer,
                length_penalty_weight=self.length_penalty_weight)
            output_beam = tf.contrib.seq2seq.dynamic_decode(
                beam_search_decoder, 
    #             impute_finished=True, # cannot be used with Beamsearch
                maximum_iterations=self.max_iterations,
                swap_memory=True)         
            sample_ids_beam = output_beam[0].predicted_ids[:, :, 0]
            final_lengths_beam = output_beam[2][:, 0]
        return (sample_ids_beam, final_lengths_beam)
    
    def decode_train(self, target, target_length, embedding,
                     decoder_cell, decoder_initial_state, 
                     output_layer=None):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_target = tf.nn.embedding_lookup(embedding, target)
        # training helper (for teacher forcing)
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            helper_train = tf.contrib.seq2seq.TrainingHelper(
                embedded_target[:, :(-1), :], # get rid of last token
                target_length - 1) # the length is thus decreased by 1
#             helper_train = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
#                 embedded_target, target_length, 
#                 embedding, self.sampling_prob)
            (decoder_outputs_train, _) = decode(
                decoder_cell, helper_train,
                decoder_initial_state, self.max_iterations, 
                output_layer=output_layer)
            (logits, _) = decoder_outputs_train

            decoder_mask_float = self.create_decoder_mask(target, target_length)[:, 1:]
            
            loss_ML = tf.contrib.seq2seq.sequence_loss(
                logits, target[:, 1:], decoder_mask_float, # get rid of start_token
                average_across_timesteps=False,
                average_across_batch=False)

            # If loss is less than average loss, then we do not train on it
            if self.clip_loss:
                assign_avg_loss = tf.assign(
                    self.avg_loss,
                    self.decay_ratio * self.avg_loss + (1 - self.decay_ratio) * tf.reduce_sum(loss_ML) / tf.reduce_sum(decoder_mask_float))
                
                """Todo: for getting PPL, this may not work"""
                with tf.control_dependencies([assign_avg_loss]):
                    clip_mask = tf.cast(
                        tf.greater(loss_ML, self.avg_loss * self.clip_ratio),
                        tf.float32)

                    decoder_mask_float = decoder_mask_float * clip_mask
                    loss_ML = loss_ML * clip_mask
            
            num_tokens = tf.reduce_sum(decoder_mask_float)    
            total_loss = tf.reduce_sum(loss_ML)
        
        return (loss_ML, num_tokens, total_loss)
            
    def encode(self, source, source_length, embedding):
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedded_source = tf.nn.embedding_lookup(embedding, source)
            
            if self.continuous_label:
                embedding_label = tf.get_variable(
                    "embedding_label", shape=[1, self.embedding_size], dtype=tf.float32)
                # this should be of shape [batch_size * embedding_size]
                embedded_score = tf.expand_dims(self.score, axis=1) * embedding_label
#                 embedded_label = tf.expand_dims(source[1:, :])
                embedded_source = tf.concat(
                    [tf.expand_dims(embedded_score, axis=1), embedded_source], axis=1)

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            [cell_fw, cell_bw] = [
                create_MultiRNNCell( 
                    [self.hidden_size // 2] * self.num_layers, # // 2 for bi-directional
                    self.keep_prob, num_proj=None,
                    reuse=tf.AUTO_REUSE)
                for _ in range(2)]
            (encoder_output_original, encoder_final_state_original) = bidirecitonal_dynamic_lstm(
                cell_fw, cell_bw, embedded_source, source_length)
            [encoder_output, encoder_lengths, encoder_final_state] = tf.cond(
                self.is_training,
                lambda: [encoder_output_original, 
                         source_length,
                         encoder_final_state_original],
                lambda: [tf.contrib.seq2seq.tile_batch(encoder_output_original, self.beam_width),
                         tf.contrib.seq2seq.tile_batch(source_length, self.beam_width),
                         tile_multi_cell_state(encoder_final_state_original)]) # only works for decoder that has >1 layers!
        return (encoder_output, encoder_lengths, encoder_final_state)
    
    def create_placeholders(self):
#         if self.str_input:
#             assert self.vocab is not None
#             assert len(self.vocab) == self.vocab_size
            
#             self.source = tf.placeholder(
#                 tf.string, shape=[self.batch_size], name="source")
#             self.target = tf.placeholder(
#                 tf.string, shape=[self.batch_size], name="target")
            
#             default = "default_str"
            
#             tokenized_source = tf.sparse_tensor_to_dense(
#                 tf.string_split(self.source, 
#                                 delimiter=' ', 
#                                 skip_empty=True),
#                 default_value=default)
#             tokenized_target = tf.sparse_tensor_to_dense(
#                 tf.string_split(self.target, 
#                                 delimiter=' ', 
#                                 skip_empty=True),
#                 default_value=default)
            
#             self.table = tf.contrib.lookup.HashTable(
#                 tf.contrib.lookup.KeyValueTensorInitializer(
#                     [default] + vocab, [0] + list(range(vocab_size))), 
#                 0)
#             self.indexed_source = self.table.lookup(tokenized_source)
#             self.indexed_target = self.table.lookup(tokenized_target)
#         else:
        self.source = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name="source")
        self.target = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name="target")
        
        self.source_length = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="source_length")
        self.target_length = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="target_length")            
        
        # start_tokens vary across glimpses
        self.start_tokens = tf.placeholder_with_default(
            tf.constant([self.start_token] * self.batch_size),
            shape=[self.batch_size], name="start_tokens")
        
        self.keep_prob = tf.placeholder_with_default(
            tf.constant(1.0), shape=[], name="keep_prob")
        self.is_training = tf.placeholder_with_default(
            tf.constant(True), shape=[], name="is_training")

        # politeness score for each target
        if self.continuous_label:
            self.score = tf.placeholder(
                tf.float32,shape=[self.batch_size], name="politeness_score")
        
    def create_decoder_mask(self, target, target_length):
        sequence_mask = tf.sequence_mask(target_length)
        unk_mask = get_mask(target, self.unk_indices)
        decoder_mask = tf.logical_and(
            sequence_mask, tf.logical_not(unk_mask))
        decoder_mask_float = tf.cast(decoder_mask, tf.float32)
        return decoder_mask_float
    
    def create_output_layer(self):
        with tf.variable_scope("projection", reuse=tf.AUTO_REUSE):
#             embedding_init = tf.concat(
#                 [embedding_new.initialized_value(), 
#                  embedding_original.initialized_value()], 
#                 axis=0)
#             if self.proj_embedding:
#                 embedding_proj = tf.get_variable(
#                     "embedding_proj",
#                     shape=[self.hidden_size, self.embedding_size],
#                     initializer=tf.contrib.layers.xavier_initializer())
#                 # embedding_proj: shape=[hidden_size * embedding_size]
#                 # embedding_init: shape=[vocab_size * embedding_size] 
#                 # output_proj: shape=[hidden_size * vocab_size]                
#                 output_proj = tf.tanh(
#                     tf.matmul(embedding_proj, 
#                               embedding,
#                               transpose_a=False, transpose_b=True))
#                 output_proj_init = tf.tanh(
#                     tf.matmul(embedding_proj.initialized_value(), 
#                               embedding_init,
#                               transpose_a=False, transpose_b=True))
#             else:
#                 output_proj = tf.contrib.layers.xavier_initializer()

            output_layer = tf.layers.Dense(
                self.vocab_size,
                use_bias=False,
#                 kernel_initializer=output_proj_init,
#                 kernel_constraint=lambda _: output_proj,
#                 trainable=False)
                trainable=True)
        return output_layer
    
    def create_embedding(self):
        """
        both embeddings will be trainable
        When scoring, e.g., with politeness,
        first use untrained embedding_word2vec to lookup those indices,
        if not found, also look in new_vocab_politeness
        if not found, use politeness's unk embedding
        """
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            embedding_unk = tf.get_variable(
                "embedding_unk_seq2seq",
                shape=[1, self.embedding_size], 
                initializer=tf.contrib.layers.xavier_initializer())
            embedding_politeness_original = tf.get_variable(
                "embedding_politeness_original_seq2seq",
                shape=[self.shared_vocab_size_politeness, self.embedding_size],
                initializer=tf.constant_initializer(self.embedding_word2vec_politeness),
                trainable=True) # change to false for experiments
            embedding_politeness_new = tf.get_variable(
                "embedding_politeness_new_seq2seq",
                shape=[self.new_vocab_size_politeness, self.embedding_size], 
                initializer=tf.contrib.layers.xavier_initializer())            
            embedding_movie_original = tf.get_variable(
                "embedding_movie_original_seq2seq",
                shape=[self.shared_vocab_size_movie, self.embedding_size],
                initializer=tf.constant_initializer(self.embedding_word2vec_movie),
                trainable=True)
            embedding_movie_new = tf.get_variable(
                "embedding_movie_new_seq2seq",
                shape=[self.new_vocab_size_movie, self.embedding_size], 
                initializer=tf.contrib.layers.xavier_initializer())
            # Have to do it in this order, otherwise UNK token won't be 0
            embedding = tf.concat(
                [embedding_unk, 
                 embedding_politeness_original, embedding_politeness_new,
                 embedding_movie_original, embedding_movie_new],
                axis=0)
        
        return embedding


# In[ ]:


# import numpy as np

# tf.reset_default_graph()
# model = Seq2Seq(
#     batch_size=2,
#     new_vocab_size=3, shared_vocab_size=5,
#     embedding_word2vec=np.random.rand(5, 300), embedding_size=300,
#     hidden_size=14, num_layers=2,
#     max_iterations=11,
#     start_token=1, end_token=2, unk_indices=[0],
#     attention_size=512, attention_layer_size=256,
#     beam_width=10, length_penalty_weight=1.0,
#     gpu_start_index=0, 
#     num_gpus=1, # set to 1 when testing
#     learning_rate=0.0001, 
#     clipping_threshold=5.0,
#     feed_final_state=False,
#     sampling_prob=0.25,
#     monotonic_attention=False,
#     proj_embedding=True)


# In[ ]:


# pprint(model.trainable_variables)

