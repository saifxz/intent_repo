# main.py


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from inference import IntentInferenceEngineV2, IntentInferenceEngineV1
# import logging
import time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np
import uuid
from fastapi import Request
from logger_config import request_id_var
import os

from producer_class import QueryProducer
from qdrant_class import SemanticCache


from logger_config import Logger

logger = Logger("AppLogger")


app = FastAPI(title="E-commerce Intent Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]  # short ID

    # Set request ID in context
    request_id_var.set(request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response

@app.get("/")
def root():
    return {"message": "Intent Classifier API is running 🚀"}

try:
    logger.info("Initializing Semantic Cache...")
    cache = SemanticCache()
    logger.info("Semantic Cache initialized successfully ✅")
except Exception as e:
    # This will catch if Qdrant host is unreachable
    logger.error(f"FATAL: Could not connect to Qdrant: {e}")
    cache = None 



try:
    logger.info("Starting Model loading sequence...")
    engine_v2 = IntentInferenceEngineV2(
        model_path="models/intent_classifier_v1.joblib"
    )
    logger.info("Inference Engine V2 loaded successfully ✅")

    logger.info("Initializing RabbitMQ Producer...")
    api_producer = QueryProducer(name="api_App", engine=engine_v2)
    logger.info("API Producer (RabbitMQ) initialized successfully ✅")

except Exception as e:
    logger.error(f"CRITICAL ERROR during startup: {e}", exc_info=True)
    raise e




MODEL_PATH = os.path.abspath("bert_sentiment_model")

# MODEL_PATH = "bert_sentiment_model"
sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,local_files_only=True)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,local_files_only=True)
sentiment_model.eval()


def get_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        pred = int(np.argmax(probs))
    
    rating = pred + 1
    sentiment = "Negative" if rating <= 2 else "Neutral" if rating == 3 else "Positive"
    sentiment_probs = {
        "Negative": probs[0],
        "Neutral": probs[1],
        "Positive": probs[2]
    }
    logger.info(f"Sentiment analysis for '{text}': {sentiment} (Probs: {sentiment_probs})") 
    return sentiment, sentiment_probs

class Query(BaseModel):
    text: str

@app.post("/predict_intent_v2")
def predict_intent_v2(query: Query):
    start_time = time.time()
    logger.info(f"Received request: '{query.text}'")
    
    try:
        # 1. Cache Lookup
        sentiment, sentiment_probs = get_sentiment(query.text)


        if cache:
            cached_intent = cache.check_cache(query.text)
            logger.info(f"Cache lookup for: '{query.text}'")
            if cached_intent:
                
                try:
                    api_producer.publish_query(query_text=query.text)
                    logger.info("Message published to RabbitMQ.")
                except Exception as p_err:
                    logger.warning(f"RabbitMQ Publish failed: {p_err}")

                logger.info(f"Result served from Cache for: '{query.text}'")
                return {
                    "query": query.text,
                    "intent": cached_intent,
                    "confidence": 1.0,
                    "source": "cache"
                }
            
            logger.info(f"No cache hit for: '{query.text}'. Proceeding to inference.") 
        
        # 2. Inference
        result = engine_v2.predict(query.text)
        logger.info(f"Inference result for '{query.text}': {result}")

        top_intents = engine_v2.predict_top_k(query.text, k=5)
        logger.info(f"Top intents for '{query.text}': {top_intents}")

        intent_name = str(result.get("intent"))
        confidence = result.get("confidence")
        logger.info(f"Inference complete. Intent: {intent_name} ({confidence})")

        # 3. Update Cache if reliable
        # if result.get("is_reliable") and cache:
        #     logger.info(f"Updating cache for reliable prediction: '{query.text}'")
        cache.update_cache(query.text, intent_name)

        # 4. RabbitMQ Publish
        try:
            api_producer.publish_query(query_text=query.text)
            logger.info("Message published to RabbitMQ.")
        except Exception as p_err:
            # We log the error but don't fail the request 
            # because the user already got their answer.
            logger.warning(f"RabbitMQ Publish failed: {p_err}")

        process_time = time.time() - start_time
        logger.info(f"Request processed in {process_time:.4f}s")
        
        return {
            "query": query.text,
            "intent": intent_name,
            "confidence": confidence,
            "source": "inference_engine",
            "top_intents": top_intents,
            "sentiment": sentiment,
            "sentiment_probs": sentiment_probs
        }

    except Exception as e:
        logger.error(f"Error in V2 prediction flow: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from fastapi.middleware.cors import CORSMiddleware
# from inference import IntentInferenceEngineV2, IntentInferenceEngineV1
# import logging

# from producer_class import QueryProducer
# from qdrant_class import SemanticCache

# # ----------------------------
# # Logging Setup
# # ----------------------------
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # ----------------------------
# # App Initialization
# # ----------------------------
# app = FastAPI(title="E-commerce Intent Classifier API")

# cache = SemanticCache()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 🔒 Change to frontend URL in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # ----------------------------
# # Request Schema
# # ----------------------------
# class Query(BaseModel):
#     text: str

# # ----------------------------
# # Load Models (on startup)
# # ----------------------------
# try:
#     logger.info("Loading models...")

#     engine_v2 = IntentInferenceEngineV2(
#         model_path="models/intent_classifier_v1.joblib"
#     )

#     # engine_v1 = IntentInferenceEngineV1(
#     #     model_path="models/intent_lstm.keras",
#     #     tokenizer_path="models/tokenizer.joblib",
#     #     label_encoder_path="models/label_encoder.joblib"
#     # )


#     logger.info("Models loaded successfully ✅")

# except Exception as e:
#     logger.error(f"Error loading models: {e}")
#     raise e


# api_producer = QueryProducer(name="api_App", engine=engine_v2)
# print("API Producer initialized.")


# # ----------------------------
# # Routes
# # ----------------------------

# @app.get("/")
# def root():
#     return {"message": "Intent Classifier API is running 🚀"}


# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}




# @app.post("/predict_intent_v1")
# def predict_intent_v1(query: Query):
#     try:
#         # 1. First, check the Semantic Cache
#         cached_intent = cache.check_cache(query.text)
#         if cached_intent:
#             return {
#                 "query": query.text,
#                 "intent": cached_intent,
#                 "confidence": 1.0,
#                 "source": "cache"
#             }

#         result = engine_v2.predict(query.text)
#         intent_name = str(result.get("intent"))

#         if result.get("is_reliable"):
#             cache.update_cache(query.text, intent_name)

#         # 4. RabbitMQ and return
#         api_producer.publish_query(query_text=query.text)
        
#         return {
#             "query": query.text,
#             "intent": intent_name,
#             "confidence": result.get("confidence"),
#             "source": "inference_engine"
#         }

#     except Exception as e:
#         logger.error(f"Error: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error")

# @app.post("/predict_intent_v2")
# def predict_intent_v2(query: Query):
#     try:
#         cached_intent = cache.check_cache(query.text)
#         if cached_intent:
#             return {
#                 "query": query.text,
#                 "intent": cached_intent,
#                 "confidence": 1.0,
#                 "source": "cache"
#             }

#         result = engine_v2.predict(query.text)
#         intent_name = str(result.get("intent"))

#         if result.get("is_reliable"):
#             cache.update_cache(query.text, intent_name)

#         # 4. RabbitMQ and return
#         api_producer.publish_query(query_text=query.text)
        
#         return {
#             "query": query.text,
#             "intent": intent_name,
#             "confidence": result.get("confidence"),
#             "source": "inference_engine"
#         }

#     except Exception as e:
#         logger.error(f"Error in V2 prediction: {e}")
#         raise HTTPException(status_code=500, detail="Prediction failed")


# # try:
# #     logger.info("Loading models...")

# #     # Existing models
# #     engine_v1 = IntentInferenceEngineV1(...)
# #     engine_v2 = IntentInferenceEngineV2(...)

# #     # New Complaint Model
# #     complaint_engine = ComplaintInferenceEngine(
# #         rf_path=r"C:\Users\saiff\Downloads\rf_model.pkl",
# #         cv_path=r"C:\Users\saiff\Downloads\count_vectorizer.pkl",
# #         tfidf_path=r"C:\Users\saiff\Downloads\tfidf_transformer.pkl",
# #         bert_path=r"C:\Users\saiff\Desktop\NLP\bert_sentiment_model"
# #     )

# #     logger.info("All models loaded successfully ✅")
# # except Exception as e:
# #     logger.error(f"Error loading models: {e}")
# #     raise e

# # # ----------------------------
# # # New Route
# # # ----------------------------

# # @app.post("/analyze_complaint")
# # def analyze_complaint(query: Query):
# #     try:
# #         # Perform combined Topic and Sentiment analysis
# #         result = complaint_engine.predict(query.text)
        
# #         return {
# #             "query": query.text,
# #             "predicted_topic": result["topic"],
# #             "topic_confidence": result["topic_probs"],
# #             "sentiment": result["sentiment"],
# #             "sentiment_confidence": result["sentiment_probs"]
# #         }
# #     except Exception as e:
# #         logger.error(f"Error in complaint analysis: {e}")
# #         raise HTTPException(status_code=500, detail="Analysis failed")

# # # ----------------------------
# # # Run with:
# # # uvicorn main:app --host 0.0.0.0 --port 8000
# # # ----------------------------