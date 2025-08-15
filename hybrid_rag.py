# hybrid_rag.py
from llama_model import LlamaModel
from vector_store import VectorStore
from knowledge_graph import KnowledgeGraph

class HybridRAG:
    def __init__(self, model_name="gpt2"):
        self.llama_model = LlamaModel(model_name)
        self.vector_store = VectorStore()
        self.knowledge_graph = KnowledgeGraph()
        self.documents = []

    def add_documents(self, documents):
        self.documents.extend(documents)
        self.vector_store.add_documents(documents)

    def generate_response(self, query):
        # First try direct knowledge graph lookup
        kg_result = self.knowledge_graph.query(query)
        if kg_result:
            return self._format_kg_response(query, kg_result)

        # Then try semantic search in knowledge graph
        kg_search_results = self.knowledge_graph.search(query)
        if kg_search_results:
            return self._format_kg_search_response(query, kg_search_results)

        # Finally fall back to vector retrieval
        return self._generate_vector_response(query)

    def _format_kg_response(self, query, kg_data):
        response = f"From knowledge graph:\n"
        for key, value in kg_data.items():
            if isinstance(value, dict):
                response += f"\n{key.capitalize()}:"
                for subkey, subvalue in value.items():
                    response += f"\n  - {subkey.replace('_', ' ').capitalize()}: {subvalue}"
            else:
                response += f"\n{key.replace('_', ' ').capitalize()}: {value}"
        return response

    def _format_kg_search_response(self, query, results):
        response = f"Related knowledge graph entries for '{query}':\n"
        for entity, score, data in results[:3]:  # Top 3 results
            response += f"\nEntity: {entity}\n"
            response += f"Relevance score: {score:.2f}\n"
            response += f"Type: {data['type']}\n"
            response += f"Key facts: {', '.join(f'{k}: {v}' for k, v in list(data.items())[1:4])}"
        return response

    def _generate_vector_response(self, query):
        relevant_indices = self.vector_store.query(query)
        used_docs = [self.documents[i] for i in relevant_indices]

        context = " ".join(used_docs)
        input_text = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        response = f"From document retrieval:\n"
        response += f"Used documents:\n"
        for i, doc in enumerate(used_docs, 1):
            response += f"{i}. {doc}\n"

        response += f"\nGenerated answer:\n{self.llama_model.generate_response(input_text)}"
        return response