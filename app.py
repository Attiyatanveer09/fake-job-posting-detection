import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Enforce secure artifact loading at absolute top level
try:
    production_model = joblib.load('best_model.pkl')
    production_tfidf = joblib.load('tfidf.pkl')
    print("✨ Production artifacts loaded and locked into memory!")
except Exception as artifact_error:
    production_model = None
    production_tfidf = None
    print(f"❌ Critical initialization failure: {artifact_error}")

def clean_text_for_production(text):
    if not text:
        return ""
    # Strip HTML markup blocks to maintain data structure hygiene
    return re.sub(r'<.*?>', '', text)

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    result_class = ""
    confidence = ""
    
    if request.method == "POST":
        try:
            job_text = request.form.get("job_text", "")
            
            # Check 1: Enforce mandatory character presence
            if not job_text or not job_text.strip():
                result = "⚠️ Please enter a job description to analyze."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # Check 2: Validation on text volumetric length (Minimum 15 words)
            word_count = len(job_text.strip().split())
            if word_count < 15:
                result = "⚠️ Please provide a detailed job description (minimum 15 words)."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)
            
            # Check 3: Domain Terminology Content Check
            job_keywords = [
                'job', 'work', 'salary', 'experience', 'company', 'role', 'apply', 'skills', 
                'position', 'hiring', 'management', 'team', 'required', 'qualification', 
                'responsibilities', 'duty', 'shift', 'office', 'remote', 'earn', 'pay',
                'engineer', 'developer', 'seeking', 'requirements', 'benefits', 'candidate',
                'designing', 'developing', 'testing', 'maintaining', 'solutions'
            ]
            
            # Coding signatures to isolate non-recruitment text inputs
            system_or_code_signals = [
                'include', 'iostream', 'namespace', 'int main', 'void', 'public class', 
                'def ', 'import ', 'javascript', '<html>', 'css', 'function()', 'printf'
            ]
            
            text_lower = job_text.lower()
            job_keyword_matches = sum(1 for word in job_keywords if word in text_lower)
            has_code_signals = any(signal in text_lower for signal in system_or_code_signals)
            
            # Reject inputs failing basic recruitment semantic thresholds
            if job_keyword_matches < 1 or has_code_signals:
                result = "❌ Error: This does not look like a valid job description. Please enter a proper job posting text."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # --- ML Pipeline Path ---
            # Strict validation checking to verify model objects are bound and fitted
            if production_tfidf is None or not hasattr(production_tfidf, 'idf_') or production_model is None:
                result = "⚙️ Backend Error: Machine Learning pipeline assets are currently unavailable or unfitted."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # Extract features via synchronized token-space vector transformation
            cleaned_payload = clean_text_for_production(job_text)
            feature_vector = production_tfidf.transform([cleaned_payload])
            
            # Compute operational inference outcome class (0 = Real, 1 = Fake)
            prediction = int(production_model.predict(feature_vector)[0])
            
            # Compute classification statistical certainty scores
            if hasattr(production_model, "predict_proba"):
                prob_matrix = production_model.predict_proba(feature_vector)[0]
                confidence_score = round(float(prob_matrix[prediction]) * 100, 2)
                confidence = f"System Confidence: {confidence_score}%"
            
            # Map interface classes sequentially based on target mapping outputs
            if prediction == 1:
                result = "🚨 ALERT: FAKE JOB POSTING DETECTED!"
                result_class = "fake"
            else:
                result = "✅ SUCCESS: REAL JOB POSTING"
                result_class = "real"
                        
        except Exception as dynamic_runtime_error:
            result = f"⚙️ Backend Error: {str(dynamic_runtime_error)}"
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