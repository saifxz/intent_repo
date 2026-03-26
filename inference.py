import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class IntentInferenceEngineV1:
    def __init__(self, model_path, tokenizer_path, label_encoder_path, max_len=20):
        # 1. Load all serialized artifacts
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = joblib.load(tokenizer_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.max_len = max_len

    def predict(self, raw_text):
        # 2. Preprocess (Same steps as training)
        seq = self.tokenizer.texts_to_sequences([raw_text])
        padded = pad_sequences(seq, maxlen=self.max_len, padding='post')
        
        # 3. Inference
        preds = self.model.predict(padded, verbose=0)
        idx = np.argmax(preds, axis=1)[0]
        confidence = np.max(preds)
        
        # 4. Map ID back to String Intent (e.g., 2 -> "CHECK_BALANCE")
        intent_name = self.label_encoder.inverse_transform([idx])[0]
        
        return {
            "intent": intent_name,
            "confidence": float(confidence),
            "is_reliable": confidence > 0.70  # Thresholding for production
        }
    
import re
import joblib
from sentence_transformers import SentenceTransformer

class IntentInferenceEngineV2:
    def __init__(self, model_path, sbert_model_name='all-MiniLM-L6-v2'):
        print(f"Loading Inference Engine with {model_path}...", flush=True)
        # Load the classifier (Ensure it was trained with probability=True if using SVM)
        self.classifier = joblib.load(model_path)
        # Load SBERT for vectorization
        self.sbert = SentenceTransformer(sbert_model_name)
        
    def _preprocess(self, text):
        """Clean the text before vectorization."""
        text = text.lower()
        return re.sub(r"[^a-zA-Z0-9\s?!\']", "", text)

    def predict(self, raw_text):
        """Converts raw text to a structured intent response."""
        cleaned_text = self._preprocess(raw_text)
        
        # 1. Get the SBERT embedding
        vector = self.sbert.encode([cleaned_text])
        
        # 2. Get Probabilities (to calculate confidence)
        # Note: Your classifier must support predict_proba
        probs = self.classifier.predict_proba(vector)[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        
        # 3. Get the Intent Label
        intent_name = self.classifier.classes_[max_idx]
        
        # 4. Return identical structure to V1
        return {
            "intent": str(intent_name),
            "confidence": float(confidence),
            "is_reliable": bool(confidence > 0.70) 
        }




