# knowledge_graph.py
class KnowledgeGraph:
    def __init__(self):
        self.entities = {
            "Paris": {
                "type": "City",
                "country": "France",
                "landmarks": ["Eiffel Tower", "Louvre"],
                "famous_for": ["art", "history", "romance"],
                "language": "French"
            },
            "France": {
                "type": "Country",
                "capital": "Paris",
                "region": "Western Europe",
                "official_language": "French",
                "currency": "Euro"
            },
            "Eiffel Tower": {
                "type": "Landmark",
                "location": "Paris",
                "built": 1889,
                "height_meters": 330,
                "architect": "Gustave Eiffel"
            }
        }

    def query(self, entity):
        """Direct lookup in knowledge graph"""
        return self.entities.get(entity, {})

    def search(self, query_text, threshold=0.7):
        """Semantic search across knowledge graph entities"""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query_text)

        results = []
        for entity, data in self.entities.items():
            # Create a description string for each entity
            desc = f"{entity} is a {data['type']}. "
            for k, v in data.items():
                if isinstance(v, list):
                    desc += f"It has {k}: {', '.join(v)}. "
                else:
                    desc += f"It has {k}: {v}. "

            entity_embedding = model.encode(desc)
            similarity = np.dot(query_embedding, entity_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entity_embedding)
            )

            if similarity > threshold:
                results.append((entity, similarity, data))

        return sorted(results, key=lambda x: x[1], reverse=True)