from inference import IntentInferenceEngineV1,IntentInferenceEngineV2
import pika
import json
import time
import random
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from producer_class import QueryProducer


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
    model_path="models/intent_classifier_v1.joblib"
)

# if __name__ == "__main__":

#     while True:
#         try:
#             connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
#             channel = connection.channel()
#             channel.exchange_declare(exchange='query_router', exchange_type='direct')

#             queries = ["I need a refund", "Add this to my cart", "Is this in stock?"]

#             print("Producer loop starting...", flush=True)
#             while True:
#                 query = random.choice(queries)
#                 publish_query(channel, query)
#                 time.sleep(5)
                
#         except Exception as e:
#             print(f" [CONNECTION ERROR] {e}. Retrying in 5s...", flush=True)
#             time.sleep(5)


if __name__ == "__main__":
    # 1. Initialize your engines (as you did before)
    # engine1 = IntentInferenceEngineV1(...)
    # engine2 = IntentInferenceEngineV2(...)

    # 2. Create multiple producer instances
    web_producer = QueryProducer(name="Web_App", engine=engine2)
    print("Web Producer initialized.")
    # mobile_producer = QueryProducer(name="Mobile_App", engine=engine2)

    # 3. Define query sets
    queries = ["I need a refund", "Add this to my cart", "Is this in stock?"]
    # print("Producers ready. Starting to publish queries...")
    # To run multiple producers simultaneously, you'd usually use threading.
    # For a single-threaded test, you can just call them manually:
    try:
        while True:
            print("Publishing queries from Web Producer...", queries)
            web_producer.publish_query(random.choice(queries))

            # mobile_producer.publish_query(random.choice(queries))
            # time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down producers...")