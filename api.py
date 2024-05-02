from flask import Flask, request, jsonify
import openai
import pickle

app = Flask(__name__)

# Set your OpenAI API key here
openai.api_key = "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"

with open('embeddings/merged.pkl', 'rb') as f:
    # Load the embedding
    embedding = pickle.load(f)


@app.route('/ask', methods=['POST'])
def ask():
    if not request.json or 'prompt' not in request.json:
        return jsonify({"error": "Missing prompt"}), 400

    user_prompt = request.json['prompt']
    # Check similarity search is working
    docs = embedding.similarity_search(user_prompt)
    #convert docs to string
    docs = str(docs)

    # Define the chatbot's initial system message
    system = [{"role": "system", "content": "Put yourself in the role of a Blum Novotest GmbH Customer Service representative. Your responsibility is to assist individuals with inquiries pertaining to Blum Novotest GmbH's products. Your answers should be concise, accurate, and based solely on this information provided." + docs+ "Avoid providing unnecessary details and refrain from fabricating information. Stay focused on addressing the specific question at hand." }]

    chat = []
    user = [{"role": "user", "content": user_prompt}]
    response = openai.ChatCompletion.create(
        messages=system + chat[-20:] + user,
        model="gpt-4", top_p=0.5, stream=True
    )

    reply = ""
    for delta in response:
        if not delta['choices'][0]['finish_reason']:
            word = delta['choices'][0]['delta']['content']
            reply += word

    return jsonify({"response": reply})

if __name__ == '__main__':
    app.run(debug=True)
