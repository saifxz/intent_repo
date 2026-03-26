from inference import IntentInferenceEngineV1,IntentInferenceEngineV2
import pika
import json
import time
import random
import re
import numpy as np
from sentence_transformers import SentenceTransformer

# Load models outside the loop
# print("Loading NLP models...", flush=True)
# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
# model = joblib.load('models/intent_classifier_v1.joblib')
# sbert = SentenceTransformer('all-MiniLM-L6-v2')

engine1 = IntentInferenceEngineV1(
    model_path="models/intent_classifier_v1.joblib",
    tokenizer_path="models/tokenizer.joblib",
    label_encoder_path="models/label_encoder.joblib"
)
engine2 = IntentInferenceEngineV2(
    model_path="models/intent_classifier_v2.joblib"
)

def publish_query(channel, query_text):
    try:

        intent = engine1.predict_intent(query_text)

        message = {"query": query_text, "category": intent}
        channel.basic_publish(
            exchange='query_router',
            routing_key=str(intent), 
            body=json.dumps(message)
        )
        print(f" [SUCCESS] Sent intent '{intent}' for: {query_text}", flush=True)
    except Exception as e:
        print(f" [ERROR in publish_query] {e}", flush=True)



if __name__ == "__main__":

    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
            channel = connection.channel()
            channel.exchange_declare(exchange='query_router', exchange_type='direct')
            
            queries = ["I need a refund", "Add this to my cart", "Is this in stock?"]
            
            print("Producer loop starting...", flush=True)
            while True:
                query = random.choice(queries)
                publish_query(channel, query)
                time.sleep(5)
                
        except Exception as e:
            print(f" [CONNECTION ERROR] {e}. Retrying in 5s...", flush=True)
            time.sleep(5)