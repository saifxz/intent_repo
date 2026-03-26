import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import torch
import numpy as np
import joblib
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class IntentInferenceEngineV1:
    """TensorFlow/Keras based Engine."""
    def __init__(self, model_path, tokenizer_path, label_encoder_path, max_len=20):
        # FIX: Ensure model_path ends in .keras or .h5 for Keras load_model
        # If your file is actually a joblib file, you MUST use joblib.load
        try:
            self.model = tf.keras.models.load_model(model_path)
        except Exception:
            print(f"Fallback: Loading {model_path} via Joblib...")
            self.model = joblib.load(model_path)
            
        self.tokenizer = joblib.load(tokenizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.max_len = max_len

    def predict(self, raw_text):
        seq = self.tokenizer.texts_to_sequences([raw_text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        preds = self.model.predict(padded, verbose=0)
        idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)
        intent_name = self.label_encoder.inverse_transform([idx])[0]
        
        return {"intent": intent_name, "confidence": float(confidence)}

import numpy as np
import joblib
import re
import spacy
from sentence_transformers import SentenceTransformer

# Load spacy model globally to avoid reloading inside the class
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# --- Preprocessing Chain of Responsibility ---

class PreprocessingHandler:
    def __init__(self):
        self._next_handler = None

    def set_next(self, handler):
        self._next_handler = handler
        return handler

    def handle(self, text: str) -> str:
        if self._next_handler:
            return self._next_handler.handle(text)
        return text

class CleanTextHandler(PreprocessingHandler):
    def handle(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9\s?!\']", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return super().handle(text)

class LemmatizeHandler(PreprocessingHandler):
    def __init__(self):
        super().__init__()
        # Ensure we keep words critical for intent (negations)
        self.stop_words = nlp.Defaults.stop_words - {"not", "no", "n't", "never"}

    def handle(self, text: str) -> str:
        doc = nlp(text)
        tokens = [
            token.lemma_ for token in doc 
            if token.text not in self.stop_words and not token.is_punct
        ]
        processed_text = " ".join(tokens)
        return super().handle(processed_text)

# --- The Inference Engine ---

class IntentInferenceEngineV2:
    """SBERT + Scikit-Learn (SVM/RF) with Chain of Responsibility Preprocessing."""
    def __init__(self, model_path, sbert_model_name='all-MiniLM-L6-v2', preprocessor=None):
        print(f"Loading Engine V2 with {model_path}...", flush=True)
        self.classifier = joblib.load(model_path)
        self.sbert = SentenceTransformer(sbert_model_name)
        
        # If no preprocessor is passed, we can define a default chain here
        if preprocessor is None:
            self.preprocessor = CleanTextHandler()
            self.preprocessor.set_next(LemmatizeHandler())
        else:
            self.preprocessor = preprocessor

    def predict(self, raw_text):
        """Processes text and predicts intent."""
        # 1. Run the preprocessing chain
        processed_text = self.preprocessor.handle(raw_text)
        
        # 2. Get the SBERT embedding
        vector = self.sbert.encode([processed_text])
        
        # 3. Get Probabilities
        probs = self.classifier.predict_proba(vector)[0]
        max_idx = np.argmax(probs)
        confidence = float(probs[max_idx])
        
        return {
            "intent": str(self.classifier.classes_[max_idx]),
            "confidence": confidence,
            "is_reliable": confidence > 0.70,
            "processed_text": processed_text # Useful for debugging in logs
        }

class ComplaintAnalysisEngine:
    """Topic Modeling + BERT Sentiment Engine."""
    def __init__(self, rf_model_path, count_vect_path, tfidf_path, bert_path):
        self.count_vect = joblib.load(count_vect_path)
        self.tfidf_transformer = joblib.load(tfidf_path)
        self.rf_model = joblib.load(rf_model_path)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained(bert_path)
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
        self.topic_mapping = {0: 'Bank Account', 1: 'Credit card', 2: 'Others', 3: 'Theft', 4: 'Loan'}

    def predict(self, text):
        # Topic
        counts = self.count_vect.transform([text])
        tfidf = self.tfidf_transformer.transform(counts)
        topic_idx = np.argmax(self.rf_model.predict_proba(tfidf)[0])
        
        # Sentiment
        inputs = self.sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.sentiment_model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
        
        return {
            "intent": self.topic_mapping[topic_idx], # Using 'intent' key for compatibility
            "sentiment": "Negative" if np.argmax(probs) == 0 else "Positive",
            "confidence": float(max(probs))
        }
    
# import re
# import joblib
# from sentence_transformers import SentenceTransformer

# class IntentInferenceEngineV2:
#     def __init__(self, model_path, sbert_model_name='all-MiniLM-L6-v2'):
#         print(f"Loading Inference Engine with {model_path}...", flush=True)
#         # Load the classifier (Ensure it was trained with probability=True if using SVM)
#         self.classifier = joblib.load(model_path)
#         # Load SBERT for vectorization
#         self.sbert = SentenceTransformer(sbert_model_name)
        
#     def _preprocess(self, text):
#         """Clean the text before vectorization."""
#         text = text.lower()
#         return re.sub(r"[^a-zA-Z0-9\s?!\']", "", text)

#     def predict(self, raw_text):
#         """Converts raw text to a structured intent response."""
#         cleaned_text = self._preprocess(raw_text)
        
#         # 1. Get the SBERT embedding
#         vector = self.sbert.encode([cleaned_text])
        
#         # 2. Get Probabilities (to calculate confidence)
#         # Note: Your classifier must support predict_proba
#         probs = self.classifier.predict_proba(vector)[0]
#         max_idx = np.argmax(probs)
#         confidence = probs[max_idx]
        
#         # 3. Get the Intent Label
#         intent_name = self.classifier.classes_[max_idx]
        
#         # 4. Return identical structure to V1
#         return {
#             "intent": str(intent_name),
#             "confidence": float(confidence),
#             "is_reliable": bool(confidence > 0.70) 
#         }



# class ComplaintAnalysisEngine:
#     def __init__(self, rf_model_path, count_vect_path, tfidf_path, bert_path):


#         self.count_vect = joblib.load(count_vect_path)
#         self.tfidf_transformer = joblib.load(tfidf_path)
#         self.rf_model = joblib.load(rf_model_path)
        
#         self.sentiment_tokenizer = AutoTokenizer.from_pretrained(bert_path)
#         self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(bert_path)
#         self.sentiment_model.eval()

#         self.topic_mapping = {
#             0: 'Bank Account services',
#             1: 'Credit card or prepaid card',
#             2: 'Others',
#             3: 'Theft/Dispute Reporting',
#             4: 'Mortgage/Loan'
#         }

#     def _clean_text(self, text):
#         text = text.lower()
#         text = re.sub(r'\[.*\]', '', text).strip()
#         text = text.translate(str.maketrans('', '', string.punctuation))
#         text = re.sub(r'\S*\d\S*\s*', '', text).strip()
#         return text.strip()

#     def predict(self, text):
#         cleaned = self._clean_text(text)

#         # 1. Topic Prediction
#         counts = self.count_vect.transform([cleaned])
#         tfidf = self.tfidf_transformer.transform(counts)
#         topic_probs_raw = self.rf_model.predict_proba(tfidf)[0]
#         topic_idx = int(np.argmax(topic_probs_raw))
        
#         topic_results = {
#             "predicted_topic": self.topic_mapping[topic_idx],
#             "topic_probabilities": {self.topic_mapping[i]: float(p) for i, p in enumerate(topic_probs_raw)}
#         }

#         # 2. Sentiment Prediction
#         inputs = self.sentiment_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
#         with torch.no_grad():
#             outputs = self.sentiment_model(**inputs)
#             probs = F.softmax(outputs.logits, dim=1).squeeze().tolist()
#             pred = int(np.argmax(probs))
        
#         rating = pred + 1
#         sentiment_label = "Negative" if rating <= 2 else "Neutral" if rating == 3 else "Positive"
        
#         sentiment_results = {
#             "sentiment": sentiment_label,
#             "sentiment_probabilities": {
#                 "Negative": probs[0],
#                 "Neutral": probs[1],
#                 "Positive": probs[2]
#             }
#         }

#         return {**topic_results, **sentiment_results}

