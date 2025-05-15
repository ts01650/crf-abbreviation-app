#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pickle
import datetime
import json
import nltk
import os
from crf_utils import buildTokenFetaures_run1

# Ensure NLTK looks only in the local nltk_data directory
local_nltk_path = os.path.join(os.path.dirname(__file__), "nltk_data")
nltk.data.path = [local_nltk_path]  # Overwrite the path list to avoid using remote

# Load CRF model
with open("crf_run1_model.pkl", "rb") as f:
    crf = pickle.load(f)

# Streamlit interface
st.title("CRF Token Classifier – Abbreviations and Long Forms")
st.markdown("Enter a biomedical sentence to classify each word as part of an abbreviation or a long form.")

# User input
user_input = st.text_area("Biomedical Input Sentence:")

# When 'Predict' is clicked
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a sentence to analyze.")
    else:
        try:
            # Tokenize input and get POS tags
            words = user_input.strip().split()
            pos_tags = [pos for _, pos in nltk.pos_tag(words)]
            
            # Extract features and make prediction
            features = buildTokenFetaures_run1(words, pos_tags)
            prediction = crf.predict([features])[0]

            # Display predictions
            st.subheader("Predictions:")
            for word, tag in zip(words, prediction):
                st.write(f"**{word}** → `{tag}`")

            # Save to log
            log_entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "input": user_input,
                "output": list(zip(words, prediction))
            }

            with open("logs.jsonl", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            st.success("Prediction complete and logged.")
