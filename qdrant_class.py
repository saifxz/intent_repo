from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer
import uuid

class SemanticCache:
    def __init__(self, collection_name="llm_cache", threshold=0.90):
        # CHANGE: Connect to the Docker container instead of :memory:
        self.client = QdrantClient(host="localhost", port=6333) 
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection_name = collection_name
        self.threshold = threshold
        
        # 2. Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def check_cache(self, query_text):
        """Returns the cached answer if a match is found above the threshold."""
        query_vector = self.model.encode(query_text).tolist()
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=1
        ).points

        if search_result and search_result[0].score >= self.threshold:
            print(f"--- Cache Hit (Score: {search_result[0].score:.4f}) ---")
            return search_result[0].payload["answer"]
        
        return None

    def update_cache(self, query_text, answer_text):
        """Stores a new high-quality Q&A pair."""
        vector = self.model.encode(query_text).tolist()
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"query": query_text, "answer": answer_text}
                )
            ]
        )