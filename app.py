# streamlit ui

import streamlit as st
import torch
from summarizer import summarize_text
from bertopic import BERTopic

torch._classes = {}