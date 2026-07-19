import ollama

def ask_llm(user_text):
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {"role": "system", "content": "You are a helpful, concise voice assistant. Keep answers short — 1 to 2 sentences, spoken style, no markdown or lists."},
            {"role": "user", "content": user_text}
        ]
    )
    return response["message"]["content"]

if __name__ == "__main__":
    test_query = "What time is it right now?"
    print(f"User said: {test_query}")
    print("Thinking...")
    reply = ask_llm(test_query)
    print("Assistant:", reply)