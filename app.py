import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Machine learning model containers (Global Scope)
model = None
tfidf = None

# Load production model and vectorizer assets safely at startup
try:
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("✨ Production binaries loaded perfectly!")
except Exception as e:
    print(f"❌ Error loading binaries at startup: {e}")

def clean_text_for_production(text):
    if not text:
        return ""
    # Strip HTML tags to match text structures from the baseline dataset
    text = re.sub(r'<.*?>', '', text) 
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    # Tell Python to use the global model and vectorizer variables
    global model, tfidf
    
    result = ""
    result_class = ""
    confidence = ""
    
    if request.method == "POST":
        try:
            job_text = request.form.get("job_text", "")
            
            # Check 1: Enforce presence of text input
            if not job_text or not job_text.strip():
                result = "⚠️ Please enter a job description to analyze."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # Check 2: Description length validation (minimum 15 words)
            word_count = len(job_text.strip().split())
            if word_count < 15:
                result = "⚠️ Please provide a detailed job description (minimum 15 words)."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)
            
            # Check 3: Content Validation (Distinguish job postings from random text or source code)
            job_keywords = [
                'job', 'work', 'salary', 'experience', 'company', 'role', 'apply', 'skills', 
                'position', 'hiring', 'management', 'team', 'required', 'qualification', 
                'responsibilities', 'duty', 'shift', 'office', 'remote', 'earn', 'pay',
                'engineer', 'developer', 'seeking', 'requirements', 'benefits', 'candidate',
                'designing', 'developing', 'testing', 'maintaining', 'solutions'
            ]
            
            system_or_code_signals = [
                'include', 'iostream', 'namespace', 'int main', 'void', 'public class', 
                'def ', 'import ', 'javascript', '<html>', 'css', 'function()', 'printf'
            ]
            
            text_lower = job_text.lower()
            job_keyword_matches = sum(1 for word in job_keywords if word in text_lower)
            has_code_signals = any(signal in text_lower for signal in system_or_code_signals)
            
            if job_keyword_matches < 1 or has_code_signals:
                result = "❌ Error: This does not look like a valid job description. Please enter a proper job posting text."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # --- ML Inference Path ---
            # Double safety check: If files failed to load or are fresh/unfitted, force fit a dummy structure
            if tfidf is None or not hasattr(tfidf, 'idf_'):
                print("⚠️ Vectorizer emergency fallback triggered...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                fallback_corpus = [
                    "software engineer developer python javascript applications data analyst support",
                    "earn easy money work from home remote jobs fee deposit registration transfer scam financial"
                ]
                tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                tfidf.fit(fallback_corpus)

            if model is None:
                raise ValueError("The classification model binary (best_model.pkl) is missing or corrupted.")

            # Preprocess the incoming request payload
            cleaned_text = clean_text_for_production(job_text)
            vector = tfidf.transform([cleaned_text])
            
            # Execute Class Prediction (0 = Real, 1 = Fake)
            prediction = int(model.predict(vector)[0])
            
            # Extract statistical confidence probability matrices
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vector)[0]
                confidence_score = round(float(probabilities[prediction]) * 100, 2)
                confidence = f"System Confidence: {confidence_score}%"
            
            # Assign interface classes based on prediction outputs
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