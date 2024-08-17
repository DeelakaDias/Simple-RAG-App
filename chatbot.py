import streamlit as st
from llama_index.llms.groq import Groq

from llama_index.core.node_parser import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

documents = SimpleDirectoryReader('./data/').load_data()
Settings.llm = Groq(
        model = "Llama3-8b-8192",
        api_key = "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",
        temperature = 0
    )

Settings.embed_model = HuggingFaceEmbedding(model_name = "BAAI.bge-small-ev-v1.5")