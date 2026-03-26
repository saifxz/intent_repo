from sentence_transformers import SentenceTransformer

# This downloads the model weights to the default cache folder
model_name = 'all-MiniLM-L6-v2' 
print(f"Downloading {model_name}...")
SentenceTransformer(model_name)
print("Download complete!")