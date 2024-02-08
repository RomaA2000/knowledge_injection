import torch
from transformers import LlamaTokenizer, LlamaModel
from datasets import load_dataset, Dataset
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

PROMPT_QUERY = "Query: {}"
PROMPT_DOCUMENT = "Document: {}"
PROMPT_CONTEXT = "Context 1: {}\nContext 2: {}\nInformation: {}"
MODEL_PATH = "workspace/Llama-2-7b-hf"
DATASET_HF_NAME = "fpb"
CSV_PATH = "tweets.csv"
COLUMN_TEXT = "text"

def load_llama_model(model_path):
    model = LlamaModel.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    return model, tokenizer

def tokenize_and_chunk(csv_path):
    df = pd.read_csv(csv_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = []
    for text in df[COLUMN_TEXT]:
        chunks.extend(text_splitter.split_text(text))
    return chunks

def embed_text(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def embed_chunks(chunks, model, tokenizer):
    embeddings = []
    for chunk in chunks:
        embedding = embed_text(PROMPT_DOCUMENT.format(chunk), model, tokenizer)
        embeddings.append(embedding)
    return np.vstack(embeddings)

def embed_queries(queries, model, tokenizer):
    query_embeddings = []
    for query in queries:
        embedding = embed_text(PROMPT_QUERY.format(query), model, tokenizer)
        query_embeddings.append(embedding)
    return np.vstack(query_embeddings)

def find_closest_contexts(query_embeddings, doc_embeddings, top_k=2):
    similarities = cosine_similarity(query_embeddings, doc_embeddings)
    closest_indices = np.argsort(similarities, axis=1)[:, -top_k:]
    return closest_indices

def create_augmented_dataset(hf_dataset, closest_indices, chunks):
    augmented_texts = []
    for i, data in enumerate(hf_dataset):
        context_1 = chunks[closest_indices[i, 0]]
        context_2 = chunks[closest_indices[i, 1]]
        new_text = PROMPT_CONTEXT.format(context_1, context_2, data[COLUMN_TEXT])
        augmented_texts.append(new_text)
    augmented_dataset = hf_dataset.map(lambda example, idx: {COLUMN_TEXT: augmented_texts[idx]}, with_indices=True)
    return augmented_dataset

def main():
    model, tokenizer = load_llama_model(MODEL_PATH)
    
    hf_dataset = load_dataset(DATASET_HF_NAME)['train']
    chunks = tokenize_and_chunk(CSV_PATH)
    
    doc_embeddings = embed_chunks(chunks, model, tokenizer)
    queries = [text for text in hf_dataset[COLUMN_TEXT]]
    query_embeddings = embed_queries(queries, model, tokenizer)
    
    closest_indices = find_closest_contexts(query_embeddings, doc_embeddings)
    augmented_dataset = create_augmented_dataset(hf_dataset, closest_indices, chunks)
    
    augmented_dataset.to_csv("augmented_dataset.csv")

if __name__ == "__main__":
    main()