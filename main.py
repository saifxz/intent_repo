# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from inference import IntentInferenceEngineV2, IntentInferenceEngineV1,ComplaintInferenceEngine
import logging

# ----------------------------
# Logging Setup
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="E-commerce Intent Classifier API")

# ----------------------------
# CORS Configuration
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 🔒 Change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Request Schema
# ----------------------------
class Query(BaseModel):
    text: str

# ----------------------------
# Load Models (on startup)
# ----------------------------
try:
    logger.info("Loading models...")

    engine_v2 = IntentInferenceEngineV2(
        model_path="models/intent_classifier_v1.joblib"
    )

    engine_v1 = IntentInferenceEngineV1(
        model_path="models/intent_lstm.keras",
        tokenizer_path="models/tokenizer.joblib",
        label_encoder_path="models/label_encoder.joblib"
    )


    logger.info("Models loaded successfully ✅")

except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e





# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return {"message": "Intent Classifier API is running 🚀"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict_intent_v1")
def predict_intent_v1(query: Query):
    try:
        
        result = engine_v1.predict(query.text)
        print(f"V1 Prediction for '{query.text}': {result}", flush=True)
        return {
            "query": query.text,
            "intent": str(result.get("intent")),
            "confidence": float(result.get("confidence", 0)),
            "is_reliable": bool(result.get("is_reliable", False))
        }

    except Exception as e:
        logger.error(f"Error in V1 prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_intent_v2")
def predict_intent_v2(query: Query):
    try:
        result = engine_v2.predict(query.text)
        print(result)
        return {
            "query": query.text,
            "intent": str(result.get("intent")),
            "confidence": float(result.get("confidence", 0)),
            "is_reliable": bool(result.get("is_reliable", False))
        }

    except Exception as e:
        logger.error(f"Error in V2 prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


try:
    logger.info("Loading models...")

    # Existing models
    engine_v1 = IntentInferenceEngineV1(...)
    engine_v2 = IntentInferenceEngineV2(...)

    # New Complaint Model
    complaint_engine = ComplaintInferenceEngine(
        rf_path=r"C:\Users\saiff\Downloads\rf_model.pkl",
        cv_path=r"C:\Users\saiff\Downloads\count_vectorizer.pkl",
        tfidf_path=r"C:\Users\saiff\Downloads\tfidf_transformer.pkl",
        bert_path=r"C:\Users\saiff\Desktop\NLP\bert_sentiment_model"
    )

    logger.info("All models loaded successfully ✅")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise e

# ----------------------------
# New Route
# ----------------------------

@app.post("/analyze_complaint")
def analyze_complaint(query: Query):
    try:
        # Perform combined Topic and Sentiment analysis
        result = complaint_engine.predict(query.text)
        
        return {
            "query": query.text,
            "predicted_topic": result["topic"],
            "topic_confidence": result["topic_probs"],
            "sentiment": result["sentiment"],
            "sentiment_confidence": result["sentiment_probs"]
        }
    except Exception as e:
        logger.error(f"Error in complaint analysis: {e}")
        raise HTTPException(status_code=500, detail="Analysis failed")

# ----------------------------
# Run with:
# uvicorn main:app --host 0.0.0.0 --port 8000
# ----------------------------