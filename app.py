from flask import Flask, request, jsonify, render_template
from models.chatbot_model import chatbot_response

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_response():
    try:
        user_text = request.json.get("msg")
        response = chatbot_response(user_text)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in /get{e}")
        return jsonify({"response": "Üzgünüm, bir hata oluştu."}),400

if __name__ == "__main__":
    app.run(debug=True)
