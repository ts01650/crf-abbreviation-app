#!/usr/bin/env python
# coding: utf-8




import streamlit as st
import pickle
import datetime
import json
import nltk
from crf_utils import buildTokenFetaures_run1
nltk.download('averaged_perceptron_tagger')

# Load CRF model
with open("crf_run1_model.pkl", "rb") as f:
    crf = pickle.load(f)

st.title("CRF Token Classifier – Abbreviations and Long Forms")

user_input = st.text_area("Enter a biomedical sentence:")

if st.button("Predict"):
    words = user_input.strip().split()
    pos_tags = [pos for _, pos in nltk.pos_tag(words)]
    features = buildTokenFetaures_run1(words, pos_tags)
    prediction = crf.predict([features])[0]

    for word, tag in zip(words, prediction):
        st.write(f"{word} → **{tag}**")

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "input": user_input,
        "output": list(zip(words, prediction))
    }

    with open("logs.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")


# In[ ]:




