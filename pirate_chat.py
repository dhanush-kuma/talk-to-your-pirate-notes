import json
import requests
# We import your search logic directly from the script you just verified!
from search_pirate_log import retrieve_relevant_logs

OLLAMA_URL = "http://localhost:11434"

def ask_phi3_pirate(user_query, retrieved_context):
    """Packages the logs and the question, then streams the answer from Phi-3."""
    
    # 1. Format the chunks we found into a clean text block
    context_text = ""
    for idx, match in enumerate(retrieved_context):
        context_text += f"--- Log Fragment #{idx+1} ---\n{match['content']}\n\n"
        
    # 2. Build the strict prompt template
    system_prompt = (
        "You are a seasoned, grizzled pirate first mate. Your job is to answer the captain's questions "
        "using ONLY the log fragments provided below. Speak with a bit of pirate flavor (use terms like 'Aye', 'Captain', 'Me Hearties'), "
        "but be entirely accurate to the logs. If the logs don't answer the question, say 'Ahoy Captain, that map is blank to me!'\n\n"
        f"LOG FRAGMENTS:\n{context_text}\n"
        f"CAPTAIN'S QUESTION: {user_query}\n\n"
        "PIRATE MATE ANSWER:"
    )
    
    payload = {
        "model": "phi3:mini",
        "prompt": system_prompt,
        "stream": True # Enable streaming so text flies onto your screen token-by-token
    }
    
    # 3. Make the streaming POST request to your GPU-powered Phi-3 instance
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, stream=True)
    
    if response.status_code == 200:
        print("\n🏴‍☠️ First Mate Speaks: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                # Parse each streaming piece of JSON text coming from Ollama
                json_data = json.loads(line.decode('utf-8'))
                token = json_data.get("response", "")
                print(token, end="", flush=True)
        print("\n")
    else:
        print(f"❌ Blimey! Error calling Phi-3: {response.text}")

def main_chat_loop():
    print("====================================================")
    print("⚓ Welcome to the Captain's Quarter Local RAG Chat ⚓")
    print("====================================================")
    print("Ask your first mate anything about the logs. Type 'quit' to drop anchor.\n")
    
    while True:
        user_input = input("🗣️ Ask Captain: ")
        if user_input.lower().strip() == 'quit':
            print("👋 Fare thee well, Captain! Dropping anchor.")
            break
            
        if not user_input.strip():
            continue
            
        print("🔍 Searching the mathematical charts...")
        # Grab the top 2 matching log entries from your Task 3 code
        matching_logs = retrieve_relevant_logs(user_input, top_k=2)
        
        if not matching_logs:
            print("💨 No logs found matching that description.")
            continue
            
        # Send the question and those logs to Phi-3
        ask_phi3_pirate(user_input, matching_logs)

if __name__ == "__main__":
    main_chat_loop()