"""
code requires that GoogleNews-vectors-negative300.bin is in the same directory
This code goes over the first pass, performing two things:
    1. pre-process all requests
    2. turn scores to binary label according to Danescu-Niculescu-Mizil et al. 2013
"""

import tensorflow as tf
import numpy as np
import csv
import os
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Stanford Politeness Corpus")
    parser.add_argument(
        "--wiki_file", type=str, default="wikipedia.annotated.csv",
        help="path to WIKI politeness data file")
    parser.add_argument(
        "--se_file", type=str, default="stack-exchange.annotated.csv",
        help="path to SE politeness data file")
    args = parser.parse_args()
    return args


current_dir = "data/Stanford_politeness_corpus/"
args = parse_args()
file_names = [args.wiki_file, args.se_file]


n_header_lines = 1 # number of header lines in the .csv file

for file_name in file_names:
    file_paths = [os.path.join(current_dir, file_name)]
    request_lst = [] 
    score_lst = []
    unlabeled_request_lst = []
    
    with tf.Graph().as_default():
        # read data
        filename_queue = tf.train.string_input_producer(file_paths, num_epochs=1, shuffle=False)
        reader = tf.TextLineReader(skip_header_lines=n_header_lines) # skip the header lines
        (_, csv_row) = reader.read(filename_queue)

        # Default values, in case of empty columns. Also specifies the type of the
        # decoded result.
        # Format: Community,Id,Request,Score1,Score2,Score3,Score4,Score5,TurkId1,TurkId2,TurkId3,TurkId4,TurkId5,Normalized Score
        record_defaults = [[''], [''], [''], [0], [0], [0], [0], [0], [''], [''], [''], [''], [''], [0.0]]
        (_, _, request, _, _, _, _, _, _, _, _, _, _, score_normalized) = tf.decode_csv(
            csv_row, record_defaults=record_defaults)

        init_local = tf.local_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_local)
            # Start populating the filename queue.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            while True:
                try:
                    (sent, score) = sess.run([request, score_normalized])
                    request_lst.append(sent.decode('UTF-8'))
                    score_lst.append(score)
                except tf.errors.OutOfRangeError:
                    print('Done reading file.')
                    break

            coord.request_stop()
            coord.join(threads)
    
    score_25 = np.percentile(score_lst, 25) # return score at 25th percentile
    score_75 = np.percentile(score_lst, 75) # return score at 75th percentile

    # (score > score_75) * 1 means: convert score that are > score_75 to 1, otherwise to 0
    combined = [(request, (score > score_75) * 1, score)
                for (request, score) 
                in zip(request_lst, score_lst)
                if (score < score_25 or score > score_75)]
    [requests, labels, scores] = list(zip(*combined))
#     data_lst = [list(requests), list(labels), list(scores)]
    data_lst = [list(requests), list(labels)]
    
    processed_file_names = [
        prefix + file_name[:(-4)] + ".pkl"
        for prefix in ["dataset_", "labels_"]]

    processed_file_paths = [
        os.path.join(current_dir, processed_file_name)
        for processed_file_name in processed_file_names]
    
    for (processed_file_path, data) in zip (processed_file_paths, data_lst):
        with open(processed_file_path, 'wb') as fp:
            pickle.dump(data, fp)
            print("Done pickling %s" % processed_file_path)

