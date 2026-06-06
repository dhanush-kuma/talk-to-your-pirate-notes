import csv
import json
import requests

OLLAMA_URL = "http://localhost:11434"
CSV_FILE_PATH = "pirate_notes/diaries.csv"
OUTPUT_JSON_PATH = "knowledge_base.json"

def get_local_embedding(text_chunk):
    """Hits the local nomic model to convert text into a 768-dimension array."""
    payload = {
        "model": "nomic-embed-text",
        "prompt": text_chunk
    }
    try:
        response = requests.post(f"{OLLAMA_URL}/api/embeddings", json=payload)
        if response.status_code == 200:
            return response.json().get("embedding")
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
    return None

def ingest_data():
    knowledge_base = []
    
    print(f"🏴‍☠️ Opening captain logs from {CSV_FILE_PATH}...")
    with open(CSV_FILE_PATH, mode='r', encoding='utf-8') as csv_file:
        # Use DictReader to easily grab columns by name
        csv_reader = csv.DictReader(csv_file)
        
        # Let's ingest the first couple of rows/stories to build our initial database
        for row in csv_reader:
            story_id = row.get("story_id")
            pirate_name = row.get("pirate_name")
            raw_prompt_text = row.get("prompt")
            
            print(f"⚓ Processing Log Entries for Story #{story_id} by Captain {pirate_name}...")
            
            # Split the massive prompt text into smaller chunks by looking for paragraphs or double newlines
            raw_chunks = raw_prompt_text.split("\n\n")
            
            for index, chunk in enumerate(raw_chunks):
                clean_chunk = chunk.strip()
                if not clean_chunk:
                    continue  # Skip empty chunks
                
                print(f"  -> Converting chunk {index+1}/{len(raw_chunks)} into map coordinates...")
                vector = get_local_embedding(clean_chunk)
                
                if vector:
                    # Save the context alongside its metadata
                    knowledge_base.append({
                        "story_id": story_id,
                        "pirate_name": pirate_name,
                        "chunk_id": index,
                        "content": clean_chunk,
                        "embedding": vector
                    })
            
            # For testing, we just need one or two full stories processed so it runs fast!
            break 

    # Save our newly minted mathematical map to a local json file
    print(f"\n Writing vector data to {OUTPUT_JSON_PATH}...")
    with open(OUTPUT_JSON_PATH, "w", encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=2)

if __name__ == "__main__":
    ingest_data()