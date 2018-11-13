#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pickle
from util import load_pickle, dump_pickle
import itertools


# In[9]:


kk = load_pickle("/playpen2/home/tongn/transfer/preprocessing/")


# In[3]:


vocab = load_pickle("../../data/MovieTriples/vocab_all.pkl")
idx2token = {i: token for i, token in enumerate(vocab)}

def to_tokens(indices):
    tokens = [idx2token[idx] for idx in indices]
    return tokens


# In[4]:


splits = ['polite', 'neutral', 'rude']


# In[7]:


for split in splits:
    file = f"/playpen2/home/tongn/polite-dialogue-generation/data/MovieTriples/{split}_movie_target.pkl"
    x = load_pickle(file)

    scores = list(zip(*x))[2]
    print(min(scores), max(scores))
    
    for i, (_, _, score) in enumerate(x):
        if split == "polite":
            if score < 0.8: 
                x[i][2] = 1.0
        elif split == "neutral":
            if score < 0.2 or score > 0.8:
                x[i][2] = 0.5
        elif split == "rude":
            if score > 0.2:
                x[i][2] = 0.0
        else:
            print("Unrecognized split.")
            raise
            
    dump_pickle(file, x)


# In[ ]:





# In[ ]:




