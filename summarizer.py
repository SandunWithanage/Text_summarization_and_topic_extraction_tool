#
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer once
model_name = "sshleifer/distilbart-cnn-12-6"
device = 0 if torch.cuda.is_available() else -1

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarize input text using a transformer model with basic chunking support for long inputs.
    """
    try:
        # Handle long text by splitting into chunks (BART max = 1024 tokens)
        inputs = tokenizer(text, return_tensors="pt", truncation=False)
        input_ids = inputs['input_ids'][0]
        max_tokens = 1024

        if len(input_ids) <= max_tokens:
            summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
            return summary[0]['summary_text']
        else:
            # Split into chunks and summarize each
            chunks = [input_ids[i:i+max_tokens] for i in range(0, len(input_ids), max_tokens)]
            summaries = []
            for chunk in chunks:
                chunk_text = tokenizer.decode(chunk, skip_special_tokens=True)
                summary = summarizer(chunk_text, max_length=max_length, min_length=min_length, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return " ".join(summaries)

    except Exception as e:
        return f"Error during summarization: {e}"

       


