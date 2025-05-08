from transformers import pipeline

# Load summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def split_text_into_chunks(text, max_chunk_chars=3500):
    """Split the text into manageable chunks for summarization"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def summarize_text(text, max_length=130, min_length=30):
    chunks = split_text_into_chunks(text)
    full_summary = []

    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
        full_summary.append(summary[0]['summary_text'])

    combined_summary = " ".join(full_summary)
    return combined_summary, len(text.split()), len(combined_summary.split()), len(chunks)
