from flask import Flask, request, render_template
import joblib
import re
import string

app = Flask(__name__)

# Load model aur vectorizer
model = joblib.load('best_model.pkl')
tfidf = joblib.load('tfidf.pkl')

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text) 
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    result_class = ""
    
    if request.method == "POST":
        job_text = request.form.get("job_text", "")
        
        # Validation 1: Word Count check
        word_count = len(job_text.strip().split())
        if word_count < 15:
            result = "⚠️ Please provide a detailed job description (minimum 15 words)."
            result_class = "warning"
        
        # Validation 2: Sanity Check (C++ ya code detection)
        else:
            job_markers = ['job', 'work', 'salary', 'experience', 'company', 'role', 'apply', 'skills']
            code_keywords = ['include', 'iostream', 'namespace', 'int main', 'void']
            
            has_job_marker = any(marker in job_text.lower() for marker in job_markers)
            is_code = any(code in job_text.lower() for code in code_keywords)
            
            if not has_job_marker and is_code:
                result = "❌ Error: This looks like source code, not a job description."
                result_class = "fake"
            else:
                # Prediction Logic
                cleaned_text = clean_text(job_text)
                vector = tfidf.transform([cleaned_text])
                prediction = model.predict(vector)[0]
                
                if str(prediction).strip() == '1':
                    result = "🚨 ALERT: FAKE JOB POSTING DETECTED!"
                    result_class = "fake"
                else:
                    result = "✅ SUCCESS: REAL JOB POSTING"
                    result_class = "real"
                
    return render_template("index.html", result=result, result_class=result_class)

if __name__ == "__main__":
    app.run(debug=True)