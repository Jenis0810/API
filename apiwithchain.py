from flask import Flask, request, jsonify, session
import openai
import pickle
from flask_session import Session
import uuid
from pymongo import MongoClient
# import datetime

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

# Configure the session to use filesystem (alternatively, you could use Redis, Memcached, etc.)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'supersecretkey'
Session(app)

# Initialize MongoDB connection
client = MongoClient("mongodb+srv://jenis:jenis0810@faq.q9ok1cm.mongodb.net/faq")
db = client.faq
ai_collection = db.ai

with open('embeddings/merged.pkl', 'rb') as f:
    # Load the embedding
    embedding = pickle.load(f)

@app.route('/ask', methods=['POST'])
def ask():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Missing prompt"}), 400

    user_prompt = request.json['prompt']
    
    # Check if a session ID was provided
    session_id = request.json['Session-ID']

    if session_id:
        # Load existing session from MongoDB
        conversation = list(ai_collection.find({"session_id": session_id}))
        if not conversation:
            # If the session ID is not found, create a new session
            session_id = str(uuid.uuid4())
            conversation = []
    else:
        # Create a new session
        session_id = str(uuid.uuid4())
        conversation = []

    # Concatenate all previous user prompts with the current prompt
    all_user_prompts = " ".join([msg['prompt'] for msg in conversation if msg['prompt']]) + " " + user_prompt

    # Perform similarity search using the concatenated user prompts
    docs = embedding.similarity_search(all_user_prompts)
    docs = str(docs)

    # Define the chatbot's initial system message
    system = [{
        "role": "system", 
        "content": (
            "Put yourself in the role of a Blum Novotest GmbH Customer Service representative. "
            "Your responsibility is to assist individuals with inquiries pertaining to Blum Novotest GmbH's products. "
            "Your answers should be concise, accurate, and based solely on this information provided: " + docs + 
            ". Avoid providing unnecessary details and refrain from fabricating information. Stay focused on addressing the specific question at hand."
        )
    }]

    # Get the response from OpenAI API using the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=system + [{"role": "user", "content": msg['prompt']} for msg in conversation[-20:]],  # Include the last 20 messages in the conversation
        top_p=0.5
    )

    reply = response.choices[0].message['content'].strip()

    # # Get the current timestamp
    # timestamp = datetime.datetime.now()

    record = {
            "session_id": session_id,
            "prompt": user_prompt,
            "response": reply
        }

        # Add the chatbot's reply to the conversation history and save it to MongoDB
    ai_collection.update_one(
        {"session_id": session_id},
        {
            "$set": record,
            "$currentDate": {"timestamp": True}
        },
        upsert=True
    )

    # Add the chatbot's reply to the conversation history and save it to MongoDB
    ai_collection.insert_one(record)

    return jsonify({"response": reply, "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True)
