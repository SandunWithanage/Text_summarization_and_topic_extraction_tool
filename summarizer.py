#
from transformers import pipeline

# Load summarizer once (e.g., outside functions if using Streamlit)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def summarize_text(text, max_length=130, min_length=30):
    # Truncate to avoid IndexError (BART = 1024 tokens, roughly ~1024*0.75 characters)
    if len(text) > 3500:
        text = text[:3500]
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']
       


