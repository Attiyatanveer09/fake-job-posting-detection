import re
import string
import joblib

try:
    model = joblib.load("best_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    print("✨ Production artifacts loaded successfully inside sandbox tester!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {e}")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

scam_text = """Earn easy money from home! Global Data Solutions is urgently hiring remote workers for simple typing and form-filling tasks. You can work anytime you want and earn up to $4,500 per week. No experience required. Pay a registration fee of $50 via bank wire transfer to get started."""

cleaned = clean_text(scam_text)
vector = tfidf.transform([cleaned])
prediction = int(model.predict(vector)[0])

if hasattr(model, "predict_proba"):
    probs = model.predict_proba(vector)[0]
    confidence = round(float(probs[prediction]) * 100, 2)
else:
    confidence = "N/A"

print("\n--- 🔍 RANDOM FOREST DEPLOYMENT TEST ---")
print(f"Prediction Class Mapping : {prediction}")
print(f"Prediction Certainty     : {confidence}%")
if prediction == 1:
    print("Evaluation Verdict       : 🚨 ALERT: FRAUDULENT POSTING DETECTED!")
else:
    print("Evaluation Verdict       : ✅ SUCCESS: SAFE JOB ADVERTISEMENT")
print("-----------------------------------------")