from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

# Load model dan vectorizer
with open("naive_bayes_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        text = request.form["text"]
        text_tfidf = vectorizer.transform([text])
        prediction = model.predict(text_tfidf)[0]
        
        result = "POSITIVE" if prediction == 1 else "NEGATIVE"
        return jsonify({"status": "success", "prediction": result})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
