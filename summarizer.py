#
from transformers import pipeline
# Load summarizer once 
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")# Load summarizer once (e.g., outside functions if using Streamlit)
#funtion

def summarize_text(text, max_length=130, min_length=30):
    # Truncate to avoid IndexError (BART = 1024 tokens, roughly ~1024*0.75 characters)
    if len(text) > 3500:
        text = text[:3500] 
        def summarize_text(text, max_length=130, min_length=30):
    # Truncate to avoid IndexError (BART = 1024 tokens, roughly ~1024*0.75 characters)
    if len(text) > 3500:
        text = text[:3500] 
         def  summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
         summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
         text summary[0]['summary_text']
    


