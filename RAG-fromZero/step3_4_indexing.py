import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from step1_2_ingestion import ingest_and_clean_text, sentence_aware_chunking

# STEP 4 (PRODUCTION): REAL-WORLD VECTOR DATABASE WITH QDRANT


class QdrantVectorStore:
    """
    Production-grade Vector Store interface using local Qdrant engine.
    Stores embeddings on disk with HNSW graph indexing and payload indexing.
    """

    def __init__(self, db_path: str = "./qdrant_db", collection_name: str = "rag_collection"):
        self.collection_name = collection_name
        # Initialize embedded Qdrant engine running directly on local disk (No Docker needed)
        self.client = QdrantClient(path=db_path)

    def create_collection(self, vector_size: int = 384):
        """
        Creates or recreates a Qdrant collection configured for Cosine Similarity.
        """
        # Check if collection exists and recreate for clean ingestion
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )
        print(
            f"Created Qdrant Collection '{self.collection_name}' (Vector Dim: {vector_size})")

    def upsert_chunks(self, chunks: list, embeddings):
        """
        Inserts (upserts) text chunks and vector embeddings as PointStruct objects into Qdrant.
        """
        points = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point = PointStruct(
                id=idx,
                vector=vector.tolist(),
                payload={
                    "chunk_id": idx,
                    "text": chunk,
                    "char_length": len(chunk)
                }
            )
            points.append(point)

        # Batch upload points to the engine
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Successfully indexed {len(points)} vector points into Qdrant.")

    def close(self):
        """Python kapanmadan önce Qdrant bağlantısını güvenle kapatır."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()


# --- EXECUTION PIPELINE ---
if __name__ == "__main__":
    # Model loading locally
    file_path = "RAG-fromZero/data/paper.txt"
    embedding_model_path = "RAG-fromZero/model_files/multilingual-e5-small"

    # Run Step 1 & Step 2
    print("--- STEP 1: Reading and Cleaning Data ---")
    raw_text = ingest_and_clean_text(file_path)
    print(
        f"File successfully loaded! Total Character Count: {len(raw_text)}")
    print(f"Raw Text Sample: \"{raw_text[:120]}...\"\n")

    print("--- STEP 2: Executing Sentence-Aware Chunking ---")
    text_chunks = sentence_aware_chunking(
        raw_text, target_chunk_size=500, overlap_sentences=1)
    print(f"Generated {len(text_chunks)} sentence-aware chunks.\n")

    # Step 3: Embeddings
    print("--- STEP 3: Generating E5 Embeddings ---")
    embedding_model = SentenceTransformer(embedding_model_path)
    prefixed_chunks = [f"passage: {chunk}" for chunk in text_chunks]
    chunk_embeddings = embedding_model.encode(
        prefixed_chunks, normalize_embeddings=True)
    print("Chunk Embedding Output: ", chunk_embeddings)

    # Step 4: Indexing into Real Qdrant Database
    print("\n--- STEP 4 (UPDATED): Indexing into Real Qdrant Engine ---")
    qdrant_store = QdrantVectorStore(
        db_path="./RAG-fromZero/qdrant_db", collection_name="hydropower_docs")
    qdrant_store.create_collection(vector_size=384)
    qdrant_store.upsert_chunks(text_chunks, chunk_embeddings)

    print("\nVerification: Real Vector Database is persisted at './RAG-fromZero/qdrant_db'")

    qdrant_store.close()
    print("Qdrant connection closed clean.")
