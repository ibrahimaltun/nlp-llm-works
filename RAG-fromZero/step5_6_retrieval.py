from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# STEP 5 & 6: QUERY PROCESSING AND QDRANT RETRIEVAL
class QdrantRetriever:
    """
    Handles query vectorization and semantic search over local Qdrant collection.
    """

    def __init__(
            self,
            db_path: str = "./qdrant_db",
            collection_name: str = "hydropower_docs",
            model_name_or_path: str = "intfloat/multilingual-e5-small"
    ):

        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)

        # Load the exact same embedding model used during ingestion
        self.embedding_model = SentenceTransformer(model_name_or_path)

    def process_query(self, user_query: str):
        """
        STEP 5: Prepares query string with E5 specific prefix and generates normalized vector.
        """
        # E5 model requirement: Queries MUST start with "query: " prefix
        prefixed_query = f"query: {user_query.strip()}"

        query_vector = self.embedding_model.encode(
            prefixed_query,
            normalize_embeddings=True
        )
        return query_vector

    def search_relevant_chunks(self, user_query: str, top_k: int = 3):
        """
        STEP 6: Queries Qdrant HNSW index for the top-k most semantically similar chunks.
        """
        query_vector = self.process_query(user_query)

        # Execute similarity search using updated query_points API
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=top_k
        )

        # Format retrieve outputs from response.points
        retrieved_docs = []
        for hit in response.points:
            retrieved_docs.append({
                "score": round(float(hit.score), 4),
                "chunk_id": hit.payload["chunk_id"],
                "text": hit.payload["text"]
            })

        return retrieved_docs

    def close(self):
        """Cleanly release database lock."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()


# --- EXECUTION & TEST ---
if __name__ == "__main__":
    # Path to local embedding model
    embedding_model_path = "RAG-fromZero/model_files/multilingual-e5-small"
    qdrant_db_path = "RAG-fromZero/qdrant_db"

    # Initialize Retriever
    retriever = QdrantRetriever(
        db_path=qdrant_db_path,
        collection_name="hydropower_docs",
        model_name_or_path=embedding_model_path
    )

    # Test Query related to your hydropower anomaly detection paper
    # high_score_q = "What clustering or autoencoder techniques are used for anomaly detection in hydropower plants?"
    real_time_q = """How are multi-sensor time series data processed in real 
    time to identify operational anomalies in hydroelectric power plants?"""

    print(f"--- STEP 5: Processing User Query ---")
    print(f"User Query: \"{real_time_q}\"")

    print(f"\n--- STEP 6: Executing Qdrant Vector Search (Top-3 Chunks) ---")
    results = retriever.search_relevant_chunks(real_time_q, top_k=3)

    for idx, doc in enumerate(results, 1):
        print(
            f"\n[Result {idx}] (Similarity Score: {doc['score']} | Chunk ID: {doc['chunk_id']})")
        print(f"Text Content: \"{doc['text'][:200]}...\"")

    retriever.close()
    print("\nRetrieval completed cleanly.")
