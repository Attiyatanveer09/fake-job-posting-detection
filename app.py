import os
import re
import string
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)


model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")


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
        job_text = request.form["job_text"]

        if job_text.strip() != "":
            
            cleaned_text = clean_text(job_text)

           
            vector = tfidf.transform([cleaned_text])

           
            prediction = model.predict(vector)[0]

            
            if prediction == 1 or str(prediction).lower() == 'fake':
                result = "FAKE JOB POSTING"
                result_class = "fake"
            else:
                result = "REAL JOB POSTING"
                result_class = "real"

    return render_template(
        "index.html",
        result=result,
        result_class=result_class
    )

if __name__ == "__main__":
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)