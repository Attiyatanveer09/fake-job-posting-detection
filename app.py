import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Machine learning model containers
model = None
tfidf = None

# Load production model and vectorizer assets with error safety blocks
try:
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("✨ Production binaries loaded perfectly!")
except Exception as e:
    print(f"❌ Error loading binaries: {e}")

def clean_text_for_production(text):
    if not text:
        return ""
    # Strip HTML tags to match text structures from the baseline dataset
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
                'responsibilities', 'duty', 'shift', 'office', 'remote', 'earn', 'pay'
            ]
            
            # Flags indicative of programming code blocks or terminal scripts
            system_or_code_signals = [
                'include', 'iostream', 'namespace', 'int main', 'void', 'public class', 
                'def ', 'import ', 'javascript', '<html>', 'css', 'function()', 'printf'
            ]
            
            # Evaluate keyword thresholds and syntax flags
            text_lower = job_text.lower()
            job_keyword_matches = sum(1 for word in job_keywords if word in text_lower)
            has_code_signals = any(signal in text_lower for signal in system_or_code_signals)
            
            # Restrict inputs failing standard recruitment terminology benchmarks
            if job_keyword_matches < 2 or has_code_signals:
                result = "❌ Error: This does not look like a valid job description. Please enter a proper job posting text."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # --- ML Inference Path ---
            if tfidf is None or model is None:
                raise ValueError("Model artifacts are missing from host application storage.")

            # Preprocess the incoming request payload to match pipeline vector expectations
            cleaned_text = clean_text_for_production(job_text)
            vector = tfidf.transform([cleaned_text])
            
            # Execute Class Prediction (0 = Real, 1 = Fake)
            prediction = int(model.predict(vector)[0])
            
            # Extract statistical confidence probability matrices
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vector)[0]
                confidence_score = round(float(probabilities[prediction]) * 100, 2)
                confidence = f"System Confidence: {confidence_score}%"
            
            # Assign structural interface classes based on inference flags
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