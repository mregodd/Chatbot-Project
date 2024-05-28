from flask import Flask, render_template, request, jsonify
from models.chatbot_model import chatbot_response

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    user_text = request.json.get("msg")
    response = chatbot_response(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
