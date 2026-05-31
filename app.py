from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("best_model.pkl")
tfidf = joblib.load("tfidf.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""
    result_class = ""

    if request.method == "POST":

        job_text = request.form["job_text"]

        vector = tfidf.transform([job_text])

        prediction = model.predict(vector)

        if prediction[0] == 0:
            result = "REAL JOB POSTING"
            result_class = "real"

        else:
            result = "FAKE JOB POSTING"
            result_class = "fake"

    return render_template(
        "index.html",
        result=result,
        result_class=result_class
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)