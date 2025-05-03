# streamlit ui

import streamlit as st
import torch
from summarizer import summarize_text
from bertopic import BERTopic
from summarizer import summarize_text
from topic_extractor import extract_topics

torch._classes = {}


# === Helper function to group documents by topic ===
def get_documents_per_topic(topics, documents):
    topic_docs = {}
    for topic, doc in zip(topics, documents):
        if topic not in topic_docs:
            topic_docs[topic] = []
        topic_docs[topic].append(doc)
    return topic_docs

# === Topic extraction function ===
def extract_topics(text):
    # Split text into sentences
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Create and train the BERTopic model
    topic_model = BERTopic()
    topics, _ = topic_model.fit_transform(sentences)

    return topic_model, topics, sentences