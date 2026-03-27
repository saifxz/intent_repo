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

# engine1 = IntentInferenceEngineV1(
#     model_path="models/intent_classifier_v1.joblib",
#     tokenizer_path="models/tokenizer.joblib",
#     label_encoder_path="models/label_encoder.joblib"
# )
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
    # List of only instruction strings (10 per intent)
    queries = [
        # --- Intent: request_refund ---
        "I received a damaged item and I want my money back.",
        "How can I initiate a return for a full refund?",
        "The product quality is poor, I'd like a reimbursement.",
        "Can you check the status of my refund for order #9982?",
        "I was overcharged on my last transaction and need a refund.",
        "Is it possible to get a refund if I don't have the original receipt?",
        "I want to cancel my order and get a complete refund.",
        "The package never arrived; I'm requesting a money-back claim.",
        "I'd like to return this gift and get the funds back on my card.",
        "Where do I find the refund request form on the website?",

        # --- Intent: availability ---
        "Is the blue velvet sofa currently in stock?",
        "When will you be restocking the wireless gaming mice?",
        "Do you have any units of the 4K monitor left in the London warehouse?",
        "Is this specific model of running shoes available in size 10?",
        "Check if the limited edition vinyl is still for sale.",
        "Are there any pre-order slots available for the new console?",
        "Can I get an alert when the summer dress collection is available?",
        "Is the organic skincare kit available for international shipping?",
        "Do you still carry the vintage leather jackets in-store?",
        "How many units of this laptop are currently available?",

        # --- Intent: add_product ---
        "Please add two bottles of the lavender essential oil to my cart.",
        "I want to put the extra-large yoga mat into my shopping basket.",
        "Add this discounted coffee machine to my order list.",
        "Can you put one of these silk pillowcases in my bag?",
        "Add the stainless steel water bottle to my checkout items.",
        "I'd like to add the wool scarf to my cart before I pay.",
        "Wanna add some batteries to the cart please.",
        "Put the mechanical keyboard in my basket in white color.",
        "I need to add the kitchen knife set to my current shopping session.",
        "Add this protein powder to my cart in chocolate flavor."
    ]
    # print("Producers ready. Starting to publish queries...")
    # To run multiple producers simultaneously, you'd usually use threading.
    # For a single-threaded test, you can just call them manually:
    try:
        while True:
            # print("Publishing queries from Web Producer...", queries)
            web_producer.publish_query(random.choice(queries))

            # mobile_producer.publish_query(random.choice(queries))
            time.sleep(5)
    except KeyboardInterrupt:
        print("Shutting down producers...")