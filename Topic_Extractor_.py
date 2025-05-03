from bertopic import BERTopic
import re

def extract_topics(text):
    # More robust sentence splitting using regex (handles punctuation)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s for s in sentences if s]  # Remove any empty strings
    
    # Initialize the topic model (you can customize embedding_model if needed)
    topic_model = BERTopic()
    
    # Fit the model on sentences
    topics, _ = topic_model.fit_transform(sentences)
    
    return topic_model, topics, sentences