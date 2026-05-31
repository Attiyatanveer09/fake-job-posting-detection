import os
import re
import string
from flask import Flask, render_template, request, redirect, url_for
import joblib

app = Flask(__name__)

# Models ko load karein
try:
    model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    print("Model and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")

# Text cleaning function
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)  # HTML tags hatane ke liye
    text = text.translate(str.maketrans('', '', string.punctuation))  # Punctuation hatane ke liye
    text = re.sub(r'\d+', '', text)  # Numbers hatane ke liye
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    result_class = ""

    if request.method == "POST":
        job_text = request.form.get("job_text", "")

        if job_text.strip() != "":
            # 1. Text ko clean karein
            cleaned_text = clean_text(job_text)

            # 2. Vectorize karein
            vector = tfidf.transform([cleaned_text])

            # 3. Model se predict karein
            prediction = model.predict(vector)[0]

            # 4. Result Check (Naye Balanced SVM ke mutabik)
            if prediction == 1 or str(prediction).lower() == 'fake':
                result = "🚨 ALERT: FAKE JOB POSTING DETECTED!"
                result_class = "fake"
            else:
                result = "✅ SUCCESS: REAL JOB POSTING"
                result_class = "real"

    return render_template(
        "index.html",
        result=result,
        result_class=result_class
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)