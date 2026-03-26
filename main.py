# main.py


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from inference import IntentInferenceEngineV2, IntentInferenceEngineV1
import logging
import time

from producer_class import QueryProducer
from qdrant_class import SemanticCache

# ----------------------------
# Logging Setup
# ----------------------------
# Use a more descriptive format to see timestamps and log levels clearly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("API_Main")

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="E-commerce Intent Classifier API")

try:
    logger.info("Initializing Semantic Cache...")
    cache = SemanticCache()
    logger.info("Semantic Cache initialized successfully ✅")
except Exception as e:
    # This will catch if Qdrant host is unreachable
    logger.error(f"FATAL: Could not connect to Qdrant: {e}")
    cache = None 

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    text: str

# ----------------------------
# Load Models
# ----------------------------
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
    # Don't raise here if you want the app to still try to start, 
    # but usually, we want it to fail fast.
    raise e

# ----------------------------
# Routes
# ----------------------------

@app.get("/")
def root():
    return {"message": "Intent Classifier API is running 🚀"}

@app.post("/predict_intent_v2")
def predict_intent_v2(query: Query):
    start_time = time.time()
    logger.info(f"Received request: '{query.text}'")
    
    try:
        # 1. Cache Lookup
        if cache:
            cached_intent = cache.check_cache(query.text)
            if cached_intent:
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
            "source": "inference_engine"
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