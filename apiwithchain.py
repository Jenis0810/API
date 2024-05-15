from flask import Flask, request, jsonify, session
import openai
import pickle
from flask_session import Session
import uuid

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

# Configure the session to use filesystem (alternatively, you could use Redis, Memcached, etc.)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.secret_key = 'supersecretkey'
Session(app)

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
        # Load existing session
        session.sid = session_id
        if 'conversation' not in session:
            session['conversation'] = []
    else:
        # Create a new session
        session_id = str(uuid.uuid4())
        session.sid = session_id
        session['conversation'] = []

    # Concatenate all previous user prompts with the current prompt
    all_user_prompts = " ".join([msg['content'] for msg in session['conversation'] if msg['role'] == 'user']) + " " + user_prompt

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

    # Add the user's message to the conversation history
    session['conversation'].append({"role": "user", "content": user_prompt})

    # Get the response from OpenAI API using the conversation history
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=system + session['conversation'][-20:],  # Include the last 20 messages in the conversation
        top_p=0.5
    )

    reply = response.choices[0].message['content'].strip()

    # Add the chatbot's reply to the conversation history
    session['conversation'].append({"role": "assistant", "content": reply})

    return jsonify({"response": reply, "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True)
