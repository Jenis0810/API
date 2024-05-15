from flask import Flask, request, jsonify
import openai
import pickle
import uuid

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

with open('embeddings/merged.pkl', 'rb') as f:
    embedding = pickle.load(f)

conversations = {}
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    if 'prompt' not in data:
        return jsonify({"error": "Missing prompt"}), 400

    session_id = data.get('session_id', str(uuid.uuid4()))
    user_prompt = data['prompt']

    if session_id not in conversations:
        conversations[session_id] = []

    docs = embedding.similarity_search(user_prompt)
    docs_str = str(docs)

    if not conversations[session_id]:
        system_message = "Put yourself in the role of a Blum Novotest GmbH Customer Service representative..."
        system = [{"role": "system", "content": system_message + docs_str}]
    else:
        system = []

    conversations[session_id].append({"role": "user", "content": user_prompt})
    chat_context = conversations[session_id][-20:]

    try:
        response = openai.ChatCompletion.create(
            messages=system + chat_context,
            model="gpt-3.5-turbo-0125",
            top_p=0.5,
            stream=True
        )

        reply = []
        for message in response:
            if 'choices' in message and message['choices'][0]['finish_reason']:
                if 'delta' in message['choices'][0] and 'content' in message['choices'][0]['delta']:
                    reply.append(message['choices'][0]['delta']['content'])

        final_reply = "".join(reply)
        conversations[session_id].append({"role": "assistant", "content": final_reply})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"response": final_reply, "session_id": session_id})


if __name__ == '__main__':
    app.run(debug=True)
