# Imports for compatibility between Python 2&3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import xrange

import numpy as np
import pickle
from itertools import groupby
import csv
from pathlib import Path
import jsonlines
from nltk.tokenize import word_tokenize
import logging
from gensim.models import KeyedVectors
import itertools


def split_lst(lst, delimiter, keep_delimiter=True):
    """
    Split list into sublists based on a delimiter
    """
    if keep_delimiter:
        append = [delimiter]
    else:
        append = []
    sublists = [list(y) + append
                for x, y 
                in itertools.groupby(lst, lambda z: z == delimiter) 
                if not x]
    return sublists


def build_dict(lst1, lst2=[]):
    """
    Build dictionary based on one or two lists.
    Args:
        lst1
        lst2
    Returns:
        two dictionaries, the second one is the reverse of the first
    """
    if lst2 == []:
        key2val = {i: val for (i, val) in enumerate(lst1)}
        val2key = {val: i for (val, i) in enumerate(lst1)}
    else:
        assert len(lst1) == len(lst2), (
            "Error in building dictionary: "
            "the two lists do not have the same length")
        key2val = {key: val for (key, val) in zip(lst1, lst2)}
        val2key = {val: key for (key, val) in zip(lst1, lst2)}
        
    return (key2val, val2key)

def build_index2token(lst, reverse=False):
    if reverse:
        dictionary = {val: i for (i, val) in enumerate(lst)}
    else:
        dictionary = {i: val for (i, val) in enumerate(lst)}
    return dictionary


def have_duplicates(lst):
    """
    Note: Each element in lst needs to be hashable! (i.e., list of lists won't work)
    """
    return len(set(lst)) < len(lst)


def exists(path):
    fp = Path(path)
    return fp.is_file()


def load_word2vec_model():
    """
    Load word embedding model
    """
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', 
        level=logging.INFO)
    model_path = '/playpen/home/tongn/GoogleNews-vectors-negative300.bin'
    model = KeyedVectors.load_word2vec_format(fname=model_path, binary=True)
    return model


def pad(input_seqs, sequence_lengths, pad_token=0, pad_len=None):
    """
    Pad a batch to max_sequence_length along the second dimension
    Args:
        • input_seqs: a list of sequences
        • sequence_lengths: a list of sequence length
        • pad_token: token used for padding
        • pad_len: maximum lengths to be padded to
    Returns:
        • padded
    """
    if pad_len:
        max_length = pad_len
    else:
        max_length = max(sequence_lengths)
    padded = [input_seq + [pad_token] * (max_length - sequence_length) 
              for (input_seq, sequence_length)
              in zip(input_seqs, sequence_lengths)]
    return padded

def unzip_lst(lst):
    """
    unzip a list of tuples/lists to multiple lists
    """
    unzipped = list(zip(*lst))
    unzipped_lsts = [list(tp) for tp in unzipped]
    return unzipped_lsts

def zip_lsts(lsts):
    """
    zip a list of lists
    """
    lengths = [len(lst) for lst in lsts]
    assert len(list(set(lengths))) == 1 # assert that the lsts have the same lengths
    zipped_lst = [list(tp) for tp in list(zip(*lsts))]
    return zipped_lst

def load_pickle(filename):
    with open(filename, "rb") as fp:
        lst = pickle.load(fp)
    print("Done loading %s." % filename)
    return(lst)

def load_pickles(filenames):
    lsts = []
    for filename in filenames:
        lsts.append(load_pickle(filename))
    return lsts

def dump_pickle(filename, lst):
    with open(filename, "wb") as fp:
        pickle.dump(lst, fp)
        print("Done dumping %s." % filename)

def dump_pickles(filenames, lsts):
    for (filename, lst) in zip(filenames, lsts):
        dump_pickle(filename, lst)

def read_lines(filename, verbose=True):
    """
    Load a file line by line into a list
    """
    with open(filename, 'r') as fp:
        lines = fp.readlines()
    if verbose:
        print("Done reading file", filename)
    
    return [line.strip() for line in lines]

def write_lines(filename, lines, verbose=True):
    """
    Write a list to a file line by line 
    """
    with open(filename, 'w', encoding="utf-8") as fp:
        for line in lines:
            print(line, file=fp)
    if verbose:
        print("Done writing to file %s." % filename)

def last_occurance_index(string, char):
    return string.rfind(char)

def exists(path):
    fp = Path(path)
    return fp.is_file()

def read_jsonl(path):
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    print("Done reading", path)
    return data

def tokenize(sent):
    return (word_tokenize(sent))

def prepend(sents, token_index):
    assert [] not in sents # verify that there is no empty list in "sents"
    assert isinstance(sents[0], list)
    prepended = [[token_index] + sent for sent in sents]
    return prepended 

def append(sents, token_index):
    assert [] not in sents
    assert isinstance(sents[0], list)
    appended = [sent + [token_index] for sent in sents]
    return appended

def decode2string(index2token, indices, end_token="END_TOKEN", remove_END_TOKEN=False):
    """
    Decode a list of indices to string.
    Args:
        index2token: a dictionary that maps indices to tokens
        indices: a list of indices that correspond to tokens
        remove_END_TOKEN: boolean indicating whether to remove the "END_TOKEN" (optional)
    Returns:
        the decoded string
    """
    decoded = [index2token[index] for index in indices]
    while True:
        if remove_END_TOKEN == True and decoded != []:
            if decoded[-1] == end_token:
                del decoded[-1]
            else:
                break
        else:
            break
    return (' ').join(decoded)

def group_lst(lst, num_grouped):
    num_elements = len(lst)
    num_groups = num_elements // num_grouped
    truncated_lst = lst[:(num_grouped * num_groups)]
    return [truncated_lst[i: (i + num_grouped)] 
            for i in xrange(0, num_elements, num_grouped)]

def shuffle(lst1, lst2):
    """
    Shuffle two lists without changing their correspondences
    Args:
        lst1: list 1
        lst2: list 2
    Returns:
        The two shuffled lists
    """
    combined = list(zip(lst1, lst2))
    np.random.shuffle(combined)
    (shuffled_lst1, shuffled_lst2) = zip(*combined)
    return [list(shuffled_lst1), list(shuffled_lst2)]

def remove_duplicates(lst):
    """
    Remove duplicates from a list
    list element can be of any type (including list)
    
    Caution: the returned list is automatically sorted!
    """
    lst.sort()
    lst_without_duplicates = [x for (x, _) in groupby(lst)]
    num_removed = len(lst) - len(lst_without_duplicates)
    print("Removed %d duplicates!" % num_removed)
    return lst_without_duplicates

