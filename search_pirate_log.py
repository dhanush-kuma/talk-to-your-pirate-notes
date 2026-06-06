import json
import math
import requests

OLLAMA_URL = "http://localhost:11434"
DB_PATH = "knowledge_base.json"

def get_query_embedding(query_text):
    """Converts the user's question into map coordinates using the same model."""
    payload = {
        "model": "nomic-embed-text",
        "prompt": query_text
    }
    response = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
    return response.json().get("embedding") if response.status_code == 200 else None

def cosine_similarity(vec_a, vec_b):
    """Computes the geometric similarity score between two 768-dimension vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a * a for a in vec_a))
    magnitude_b = math.sqrt(sum(b * b for b in vec_b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot_product / (magnitude_a * magnitude_b)

def retrieve_relevant_logs(user_query, top_k=2):
    """Finds the top K most similar pirate logs matching the user's query."""
    # 1. Get vector for the question
    query_vector = get_query_embedding(user_query)
    if not query_vector:
        print("❌ Error: Failed to generate embedding for query.")
        return []

    # 2. Load our indexed knowledge base
    with open(DB_PATH, "r") as f:
        knowledge_base = json.load(f)

    # 3. Calculate similarity score for every single entry
    scored_entries = []
    for entry in knowledge_base:
        score = cosine_similarity(query_vector, entry["embedding"])
        scored_entries.append((score, entry))

    # 4. Sort entries by highest score first
    scored_entries.sort(key=lambda x: x[0], reverse=True)

    # 5. Extract the top matching results
    results = [entry for score, entry in scored_entries[:top_k]]
    return results

if __name__ == "__main__":
    print("🌊 Initializing Search Engine...")
    
    # Test Query - pick a keyword or topic from your pirate log entries!
    test_question = "Was there a mention about a fat merchant?"
    
    print(f"🔍 Searching logs for: '{test_question}'\n")
    matches = retrieve_relevant_logs(test_question, top_k=2)
    
    print(f"🏴‍☠️ Found the {len(matches)} most relevant log fragments:")
    print("=" * 60)
    for i, match in enumerate(matches):
        print(f"Match #{i+1} (Captain: {match['pirate_name']} | Story ID: {match['story_id']})")
        print(f"Content: {match['content']}")
        print("-" * 60)