from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import pandas as pd

app = Flask(__name__)
CORS(app)

# 🔥 Lazy Load Model (FIX MEMORY CRASH)
classifier = None

def get_model():
    global classifier
    if classifier is None:
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
    return classifier


# -------------------------------
# 🔹 Single Text Prediction
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data.get('text', '')

        if not text:
            return jsonify({"error": "No text provided"}), 400

        result = get_model()(text)  # ✅ changed

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# 🔹 Bulk CSV Prediction
# -------------------------------
@app.route('/bulk', methods=['POST'])
def bulk_predict():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']

        df = pd.read_csv(file)

        if 'text' not in df.columns:
            return jsonify({"error": "CSV must contain 'text' column"}), 400

        texts = df['text'].astype(str).tolist()

        results = get_model()(texts)  # ✅ changed

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# 🔹 Run App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)