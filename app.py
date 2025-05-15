#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import datetime
import json
import nltk
import os
from crf_utils import buildTokenFetaures_run1

# Add the local nltk_data path for offline POS tagging
nltk.data.path.append("./nltk_data")

# Load CRF model from pickle file
with open("crf_run1_model.pkl", "rb") as f:
    crf = pickle.load(f)

# Streamlit UI
st.title("CRF Token Classifier – Abbreviations and Long Forms")
st.markdown("Enter a biomedical sentence and get the abbreviation or long-form token tags.")

user_input = st.text_area("Biomedical Input Sentence:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a sentence to analyze.")
    else:
        # Tokenize input and get POS tags
        words = user_input.strip().split()
        pos_tags = [pos for _, pos in nltk.pos_tag(words)]
        
        # Extract features and predict using CRF model
        features = buildTokenFetaures_run1(words, pos_tags)
        prediction = crf.predict([features])[0]

        # Show tagged output
        st.subheader("Predictions:")
        for word, tag in zip(words, prediction):
            st.write(f"**{word}** → `{tag}`")

        # Log interaction
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input": user_input,
            "output": list(zip(words, prediction))
        }

        with open("logs.jsonl", "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        st.success("Prediction complete. Logged interaction.")
