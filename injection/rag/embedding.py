from typing import List
from langchain.embeddings.base import Embeddings
import numpy as np
import requests
import os


class LLamaEmbedding(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 16
        embeddings = []
        
        
        for idx in range(0, len(texts), batch_size):
            text_batch = texts[idx:idx+batch_size]
            instances = []
            
            for text in text_batch:
                instances.append({
                    "instruction": "Document: ",
                    "text": text
                })
            
            
            response = requests.post(
                url=os.getenv("LLAMA_URL") + "/embed",
                json={
                    "instances": instances
                },
                verify=False
            )

            response.raise_for_status()
            embeddings += response.json()["predictions"]
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]