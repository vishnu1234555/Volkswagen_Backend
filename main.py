import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# Explicitly enable CORS
CORS(app)

# Initialize Sentence Transformer model for embeddings (all-MiniLM-L6-v2)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the Qdrant client
qdrant_client = QdrantClient(url="http://localhost:6333") 

# Initialize the Groq client, pulling key from environment
groq_api_key = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key)

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Parse the JSON payload from the request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request payload."}), 400
            
        user_message = data['message']
        
        # 1. GENERATE EMBEDDING FOR USER MESSAGE
        query_vector = embedding_model.encode(user_message).tolist()
        
        # 2. QUERY QDRANT FOR CONTEXT
        # Querying the volkswagen_collection using the generated vector
        search_result = qdrant_client.search(
            collection_name="volkswagen_collection",
            query_vector=query_vector,
            limit=3
        )
        
        # Extract the text chunks from the payload
        contexts = [hit.payload.get("text", "") for hit in search_result if hit.payload and "text" in hit.payload]
        retrieved_context = "\n".join(contexts)
        
        # 3. SEND CONTEXT AND MESSAGE TO GROQ API
        system_prompt = (
            "You are a helpful and knowledgeable Volkswagen AI assistant. "
            "Use the provided context to answer the user's questions accurately. "
            "If the answer is not in the context, say that you don't know."
        )
        
        combined_system_message = f"{system_prompt}\n\nContext:\n{retrieved_context}"
        
        # Trigger actual Groq completion using llama3-8b-8192
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": combined_system_message},
                {"role": "user", "content": user_message}
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=1024,
        )
        
        # Extract the actual textual response
        groq_response_text = chat_completion.choices[0].message.content

        # 4. RETURN RESPONSE TO CLIENT
        return jsonify({
            "response": groq_response_text
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        # Return proper error details if something fails 
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("Starting Volkswagen RAG Flask Backend on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)
