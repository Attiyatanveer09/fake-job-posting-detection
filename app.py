import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

model = None
tfidf = None

try:
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("✨ Balanced Model & Vectorizer Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")

def clean_text_for_production(text):
    if not text:
        return ""
    text = re.sub(r'<.*?>', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

def is_valid_job_description(text):
    word_count = len(text.strip().split())
    if word_count < 15:
        return False, "short"

    code_signals = ['#include', 'int main', 'def ', 'import ', 'class ',
                    'public static', 'console.log', '<?php', 'SELECT * FROM']
    if sum(1 for sig in code_signals if sig in text) >= 2:
        return False, "code"

    job_vocab = ['job', 'work', 'salary', 'experience', 'company', 'role',
                 'apply', 'skills', 'position', 'team', 'responsibilities',
                 'requirements', 'hire', 'candidate', 'office', 'remote',
                 'full-time', 'part-time', 'contract', 'benefits', 'degree',
                 'earn', 'hiring', 'opportunity', 'income', 'employment',
                 'vacancy', 'opening', 'applicant', 'qualification', 'payment']
    matches = sum(1 for word in job_vocab if word in text.lower())
    if matches < 2:
        return False, "not_job"

    return True, "ok"

def check_red_flags(text):
    red_flags = [
        'bank account', 'banking details', 'banking information',
        'registration fee', 'processing fee', 'verification fee',
        'wire transfer', 'western union', 'send money',
        'no experience needed', 'no experience required',
        'no interview', 'no qualifications',
        'guaranteed income', 'guaranteed earnings',
        'earn $', 'earn up to $',
        'pre-selected', 'randomly selected', 'you have been selected',
        'unlimited income', 'instant approval',
        'provide your bank', 'financial information',
        'pay a fee', 'pay a small', 'one-time fee',
        'work from home and earn', 'make thousands',
        'immediate placement', 'approval is guaranteed',
        'no education', 'positions are filling',
        'guaranteed', 'selected for', 'congratulations',
        'provide your', 'send your', 'submit your',
        'transfer a', 'pay a registration', 'pay a processing',
        'income potential', 'earning potential',
        'one hour per day', 'working one hour',
        'no training required', 'no interview required'
    ]

    text_lower = text.lower()
    triggered = [flag for flag in red_flags if flag in text_lower]
    return triggered

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    result_class = ""
    confidence = ""

    if request.method == "POST":
        try:
            job_text = request.form.get("job_text", "")

            if not job_text or not job_text.strip():
                result = "⚠️ Please enter a job description to analyze."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            valid, reason = is_valid_job_description(job_text)

            if not valid:
                if reason == "short":
                    result = "⚠️ Please provide a detailed job description (minimum 15 words)."
                elif reason == "code":
                    result = "❌ This looks like source code, not a job description. Please paste a real job posting."
                elif reason == "not_job":
                    result = "⚠️ This doesn't appear to be a job description. Please paste a real job posting."
                result_class = "warning"

            else:
                if tfidf is None:
                    raise ValueError("Vectorizer (tfidf.pkl) failed to load at startup.")
                if model is None:
                    raise ValueError("Model (best_model.pkl) failed to load at startup.")

                # ML Prediction
                cleaned_text = clean_text_for_production(job_text)
                vector = tfidf.transform([cleaned_text])
                prediction = int(model.predict(vector)[0])

                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(vector)[0]
                    confidence_score = round(float(probabilities[prediction]) * 100, 2)
                    confidence = f"System Confidence: {confidence_score}%"

                # Red Flag Rule Layer
                triggered_flags = check_red_flags(job_text)

                if len(triggered_flags) >= 2:
                    prediction = 1
                    confidence = f"System Confidence: 99.0% (Red Flags Detected: {len(triggered_flags)})"

                if prediction == 1:
                    result = "🚨 ALERT: FAKE JOB POSTING DETECTED!"
                    result_class = "fake"
                else:
                    result = "✅ SUCCESS: REAL JOB POSTING"
                    result_class = "real"

        except Exception as err:
            result = f"⚙️ Backend Error: {str(err)}"
            result_class = "warning"

    return render_template(
        "index.html",
        result=result,
        result_class=result_class,
        confidence=confidence
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)