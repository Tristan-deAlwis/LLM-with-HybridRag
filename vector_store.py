import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(384)  # Dimension for all-MiniLM-L6-v2

    def add_documents(self, documents):
        embeddings = self.embedding_model.encode(documents)
        self.index.add(embeddings)

    def query(self, query_text, k=5):
        query_embedding = self.embedding_model.encode([query_text])
        distances, indices = self.index.search(query_embedding, k)
        return indices[0]