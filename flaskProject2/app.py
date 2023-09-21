from flask import Flask, request, render_template, jsonify, session
from gpt4all import GPT4All
import time

app = Flask(__name__)
app.secret_key = 'a_random_secret_key'  # Replace this with a secure secret key

# Load the model once when the server starts
model = GPT4All("D:\LLMs\ggml-model-gpt4all-falcon-q4_0.bin")


@app.route('/')
def chat():
    session.clear()  # Clear any existing conversation history
    return render_template('form.html')


@app.errorhandler(Exception)
def handle_exception(e):
    return str(e), 500

@app.route('/', methods=['POST'])
def chat_post():
    data = request.get_json()
    text = data['text']

    # Append the user's message to the conversation history
    if 'history' not in session:
        session['history'] = []
    session['history'].append({'role': 'user', 'content': text})

    start_time = time.time()
    # Prepare the context for the model using the chat_session context manager
    with model.chat_session():
        for message in session['history']:
            model.generate(prompt=message['content'], temp=0)  # Providing prior interactions as context
        output = model.generate(prompt=text, temp=0, max_tokens=2048)  # Generating response for the current message

    execution_time = round(time.time() - start_time, 2)  # Rounding off to two decimal places

    # Append the model's response to the conversation history
    session['history'].append({'role': 'assistant', 'content': output})

    return jsonify({"response": output, "execution_time": execution_time})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
