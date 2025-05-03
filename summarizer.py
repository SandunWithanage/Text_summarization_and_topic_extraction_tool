#
from transformers import pipeline
# Load summarizer once 
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
