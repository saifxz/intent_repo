import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from sentence_transformers import SentenceTransformer

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SemanticCache")

class SemanticCache:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Initializing Singleton SemanticCache Instance...")
            cls._instance = super(SemanticCache, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, collection_name="llm_cache", threshold=0.90, host="qdrant"):
        if self._initialized:
            return
        try:
            self.client = QdrantClient(host=host, port=6333)
            # Heavy model loading happens only once here
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.collection_name = collection_name
            self.threshold = threshold
            
            self._ensure_collection()
            self._initialized = True
            logger.info(f"SemanticCache initialized with collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to initialize SemanticCache: {e}")
            raise

    def _ensure_collection(self):
        """Checks and creates the Qdrant collection if missing."""
        if not self.client.collection_exists(self.collection_name):
            logger.info(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def check_cache(self, query_text):
        """Returns the cached answer if a match is found above the threshold."""
        try:
            query_vector = self.model.encode(query_text).tolist()
            
            search_result = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=1
            ).points

            if search_result and search_result[0].score >= self.threshold:
                logger.info(f"Cache Hit! Score: {search_result[0].score:.4f}")
                return search_result[0].payload["answer"]
            
            logger.info("Cache Miss.")
            return None
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    def update_cache(self, query_text, answer_text):
        """Stores a new high-quality Q&A pair."""
        try:
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
            logger.info(f"Cache updated for query: {query_text[:30]}...")
        except Exception as e:
            logger.error(f"Error updating cache: {e}")