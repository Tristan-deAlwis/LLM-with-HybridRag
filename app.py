# app.py
from flask import Flask, request, jsonify
from hybrid_rag import HybridRAG

app = Flask(__name__)

# Initialize the HybridRAG system
hybrid_rag = HybridRAG()

# Sample documents
sample_docs = [
    "The capital of France is Paris.",
    "Paris is known as the City of Light.",
    "The Eiffel Tower was built in 1889.",
    "France is a country in Western Europe with a rich history.",
    "The Louvre Museum in Paris houses the Mona Lisa."
]

# Add documents to the vector store
hybrid_rag.add_documents(sample_docs)

@app.route("/generate", methods=["POST"])
def generate():
    query = request.json["query"]
    response = hybrid_rag.generate_response(query)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)