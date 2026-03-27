import requests
import time

URL = "http://localhost:5000/predict_intent_v2"


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

queries = [
    "I want to return my order",
    "Where is my shipment?",
    "Payment failed but money deducted",
    "Show me latest offers",
    "Cancel my order please"
]

for i in range(10):  # number of rounds
    for q in queries:
        payload = {"text": q}
        
        try:
            start = time.time()
            response = requests.post(URL, json=payload)
            end = time.time()

            print(f"\nQuery: {q}")
            print("Response:", response.json())
            print(f"Time taken: {end - start:.4f}s")

        except Exception as e:
            print(f"Error: {e}")

        time.sleep(0.5)  # small delay (optional)