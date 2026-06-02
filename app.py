import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables for our machine learning artifacts
model = None
tfidf = None

# Load your fresh Random Forest model and vectorizer safely
try:
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("✨ Balanced Model & Vectorizer Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")

def clean_text_for_production(text):
    if not text:
        return ""
    # MATCH NOTEBOOK: We remove HTML tags if present, but we PRESERVE
    # capitalization, numbers, and punctuation exactly like your notebook's text format did!
    text = re.sub(r'<.*?>', '', text) 
    return text

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

            # Validation 1: Word Count check
            word_count = len(job_text.strip().split())
            if word_count < 15:
                result = "⚠️ Please provide a detailed job description (minimum 15 words)."
                result_class = "warning"
            
            # Validation 2: Source code check
            else:
                job_markers = ['job', 'work', 'salary', 'experience', 'company', 'role', 'apply', 'skills']
                code_keywords = ['include', 'iostream', 'namespace', 'int main', 'void']
                
                has_job_marker = any(marker in job_text.lower() for marker in job_markers)
                is_code = any(code in job_text.lower() for code in code_keywords)
                
                if not has_job_marker and is_code:
                    result = "❌ Error: This looks like source code, not a job description."
                    result_class = "fake"
                else:
                    if tfidf is None:
                        raise ValueError("Vectorizer (tfidf.pkl) failed to load at startup.")
                    if model is None:
                        raise ValueError("Model (best_model.pkl) failed to load at startup.")

                    # Process the incoming text stream using matched attributes
                    cleaned_text = clean_text_for_production(job_text)
                    vector = tfidf.transform([cleaned_text])
                    
                    # Predict Class (0 = Real, 1 = Fake)
                    prediction = int(model.predict(vector)[0])
                    
                    # Extract classification probabilities
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(vector)[0]
                        confidence_score = round(float(probabilities[prediction]) * 100, 2)
                        confidence = f"System Confidence: {confidence_score}%"
                    
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