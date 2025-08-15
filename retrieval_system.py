import pinecone

class RetrievalSystem:
    def __init__(self, api_key, index_name):
        pc = pinecone(api_key=api_key)
        self.index = pc.Index(index_name)

    def retrieve_documents(self, query, tokenizer):
        query_vector = tokenizer.encode(query, return_tensors="pt")
        results = self.index.query(vector=query_vector, top_k=5)
        return [result["id"] for result in results["matches"]]