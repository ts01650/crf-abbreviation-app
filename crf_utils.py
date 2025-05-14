#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from pprint import pprint
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def buildTokenFetaures_run1(words,pos_seq):
    featureForSen = []
    for i in range(len(words)):
        curr_word = words[i]
        curr_pos = pos_seq[i]

        #for current token
        feats = {'low':curr_word.lower(), 
                'suf3':curr_word[-3:],
                'caps':curr_word.isupper(),
                'title':curr_word.istitle(),
                'pos':curr_pos,
                'pos2':curr_pos[:2]}

        #for previous token
        if i>0:
            feats['prev_low'] = words[i-1].lower()
            feats['prev_pos'] = pos_seq[i-1]
        else:
            feats['start'] = True

        #for next token
        if i < len(words)-1:
            feats['next_low'] = words[i+1].lower()
            feats['next_pos'] = pos_seq[i+1]
        else:
            feats['end'] = True

        featureForSen.append(feats)
    return featureForSen


# In[ ]:




