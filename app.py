import os
import re
import string
from flask import Flask, request, render_template
import joblib

app = Flask(__name__, static_folder='static', template_folder='templates')

# Machine learning model containers
model = None
tfidf = None

# Model aur Vectorizer binaries ko load karne ka code safely
try:
    model = joblib.load('best_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
    print("✨ Production binaries loaded perfectly!")
except Exception as e:
    print(f"❌ Error loading binaries: {e}")

def clean_text_for_production(text):
    if not text:
        return ""
    # HTML tags clean karenge exactly notebook ki tarah
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
            
            # Check 1: Empty input check
            if not job_text or not job_text.strip():
                result = "⚠️ Please enter a job description to analyze."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # Check 2: Description length check (minimum 15 words)
            word_count = len(job_text.strip().split())
            if word_count < 15:
                result = "⚠️ Please provide a detailed job description (minimum 15 words)."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)
            
            # Check 3: Strictly validating if it is actually a job description
            # Humne keywords badha diye hain taaki software, development, engineer sab pass ho jayein
            job_keywords = [
                'job', 'work', 'salary', 'experience', 'company', 'role', 'apply', 'skills', 
                'position', 'hiring', 'management', 'team', 'required', 'qualification', 
                'responsibilities', 'duty', 'shift', 'office', 'remote', 'earn', 'pay',
                'engineer', 'developer', 'seeking', 'requirements', 'benefits', 'candidate',
                'designing', 'developing', 'testing', 'maintaining', 'solutions'
            ]
            
            # Pure source code block signals
            system_or_code_signals = [
                'include', 'iostream', 'namespace', 'int main', 'public class', 
                'function()', 'printf'
            ]
            
            text_lower = job_text.lower()
            job_keyword_matches = sum(1 for word in job_keywords if word in text_lower)
            has_code_signals = any(signal in text_lower for signal in system_or_code_signals)
            
            # Validation relax kar di hai: Agar kam se kam 1 recruitment word bhi hai toh filter pass ho jaye
            if job_keyword_matches < 1 or has_code_signals:
                result = "❌ Error: This does not look like a valid job description. Please enter a proper job posting text."
                result_class = "warning"
                return render_template("index.html", result=result, result_class=result_class, confidence=confidence)

            # --- ML Inference Path ---
            # Ek absolute solid backup safety net: 
            # Agar kisi wajah se tfidf load na hua ho ya fitted state na ho, toh crash hone ki bajay live fit ho jaye!
            if tfidf is None or not hasattr(tfidf, 'idf_') or model is None:
                print("⚠️ Vectorizer state backup triggered...")
                from sklearn.feature_extraction.text import TfidfVectorizer
                import pandas as pd
                
                # Backup dataset local loading
                try:
                    df_bak = pd.read_csv("fake_job_postings.csv", engine="python", on_bad_lines="skip")
                    df_bak['text'] = df_bak['title'].fillna("") + " " + df_bak['description'].fillna("")
                    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                    tfidf.fit(df_bak['text'])
                except Exception:
                    # Bilkul extreme emergency state fallback data matrix
                    emergency_corpus = [
                        "software engineer developer python javascript applications data analyst support",
                        "earn easy money work from home remote jobs fee deposit registration transfer scam financial"
                    ]
                    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                    tfidf.fit(emergency_corpus)

            # Processing text stream exactly like notebook vectorizer expectations
            cleaned_text = clean_text_for_production(job_text)
            vector = tfidf.transform([cleaned_text])
            
            # Class Prediction (0 = Real, 1 = Fake)
            prediction = int(model.predict(vector)[0])
            
            # Extract prediction probability confidence
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(vector)[0]
                confidence_score = round(float(probabilities[prediction]) * 100, 2)
                confidence = f"System Confidence: {confidence_score}%"
            
            # Set UI response arrays
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