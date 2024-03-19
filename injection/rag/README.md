
# Retrieval-Augmented Generation (RAG) with LLaMA2 7B and Qdrant

The RAG method leverages two chunks of text, each of size 512 tokens, which are appended before the task text. These chunks are selected using embedding matching based on cosine similarity with the query from the dataset. This ensures that each query is processed with relevant context.

### Example Prompt for RAG:
```
Context 1: {context}
Context 2: {context}
Information: {sentence}
```

### Experimental Variants:
- **Standard Model**: Using the pre-trained LLaMA2 7B model for retrieving relevant texts.
- **Finetuned Model**: Using the Prompt-tuned LLaMA2 7B model for retrieving relevant texts.

## Installation

To run this project, you need to install the necessary dependencies. Use the following command to install them:

```bash
pip install torch transformers datasets pandas langchain qdrant-client
```

## Usage

1. **Load and Modify LLaMA2 Model**: The script loads the LLaMA2 7B model and modifies it to use the last hidden state as the embedding.

2. **Tokenize and Chunk CSV Data**: Text data from the provided CSV file is tokenized and split into chunks of 512 tokens with overlaps of 50 tokens.

3. **Generate Embeddings**: The last hidden state of the LLaMA2 model is used to generate embeddings for each chunk and query.

4. **Upload to Qdrant**: The chunk embeddings are uploaded to the Qdrant vector search engine.

5. **Retrieve Relevant Contexts**: For each query, the closest context chunks are retrieved using Qdrant based on cosine similarity.

6. **Create Augmented Dataset**: The retrieved contexts are used to create a new dataset with augmented texts, which is saved as a CSV file.
