from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from functions.hybrid_search import relative_score_fusion
from dotenv import load_dotenv
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("Qdrant URL: %s", os.getenv("QDRANT_URL"))

class QdrantManager:
    """Manages Qdrant client and vector stores."""

    def __init__(self):
        """Initialize the QdrantManager."""
        self.qdrant_client = None
        self.vector_stores = {}

    def get_qdrant_client(self):
        """Lazy initialization for QdrantClient."""
        if self.qdrant_client is None:
            logger.info("Initializing Qdrant client...")
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                https=False,
                prefer_grpc=True,
            )
        return self.qdrant_client

    def get_vector_store(self, dataset: str, hybrid: bool = False):
        """Lazy initialization for QdrantVectorStore."""
        if dataset not in self.vector_stores:
            logger.info(f"Initializing QdrantVectorStore for dataset: {dataset}...")
            client = self.get_qdrant_client()

            # # Ensure the collection exists or create it
            # collection_info = client.get_collection(collection_name=dataset)
            # if not collection_info:
            #     client.create_collection(
            #         collection_name=dataset,
            #         vectors_config={"size": 1536, "distance": "Cosine"},  # Adjust based on your embedding size
            #     )

            self.vector_stores[dataset] = QdrantVectorStore(
                collection_name=dataset,
                client=client,  # Use synchronous client
                batch_size=20,
                prefer_grpc=True,
                enable_hybrid=hybrid,
                hybrid_fusion_fn=relative_score_fusion if hybrid else None,
            )
        return self.vector_stores[dataset]

# Initialize QdrantManager
qdrant_manager = QdrantManager()