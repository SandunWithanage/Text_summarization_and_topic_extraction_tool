# streamlit ui

import streamlit as st
import torch
from summarizer import summarize_text
from bertopic import BERTopic
from summarizer import summarize_text
from topic_extractor import extract_topics

torch._classes = {}



st.title("📚 Text Summarization & Topic Extraction Tool")

input_text = st.text_area("Paste your text here", height=300)

uploaded_file = st.file_uploader("Or upload a .txt file", type="txt")
if uploaded_file:
    input_text = uploaded_file.read().decode("utf-8")

if st.button("Summarize & Extract Topics") and input_text:
    with st.spinner("Summarizing..."):
        summary = summarize_text(input_text)
        st.subheader("📝 Summary")
        st.write(summary)

    with st.spinner("Extracting topics..."):
        model, topics, sentences = extract_topics(input_text)
        topic_info = model.get_topic_info()
        st.subheader("📌 Topics")
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:
                st.markdown(f"**Topic {row['Topic']}**: {row['Name']}")
                st.write(model.get_topic(row['Topic']))

        st.subheader("🔎 Clickable Topic Explorer")
        topic_docs = model.get_documents_per_topic()
        for topic, docs in topic_docs.items():
            if topic != -1:
                with st.expander(f"Topic {topic}"):
                    for doc in docs[:5]:  # Show top 5 sentences
                        st.write(f"- {doc}")
