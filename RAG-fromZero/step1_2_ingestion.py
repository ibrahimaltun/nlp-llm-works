import os
import re

# STEP 1: DATA INGESTION AND CLEANING


def ingest_and_clean_text(file_path: str) -> str:
    """
    Reads a raw text file using UTF-8 encoding and normalizes
    unwanted white spaces, line breaks, and system noise.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Source file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        raw_text = file.read()

    # Remove excessive newlines and tab spaces, replacing them with single spaces
    cleaned_text = re.sub(r'\s+', ' ', raw_text).strip()
    return cleaned_text

# STEP 2: SENTENCE-AWARE SEMANTIC CHUNKING


def sentence_aware_chunking(text: str, target_chunk_size: int = 500, overlap_sentences: int = 1) -> list:
    """
    Splits raw text into natural sentences using regex punctuation matching.
    Accumulates sentences up to target_chunk_size while preserving sentence boundaries
    and applying sentence-level overlap for contextual continuity.
    """
    # Regex split that preserves sentence boundaries (. ! ?)
    sentence_pattern = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_pattern.split(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        sentence_len = len(sentence)

        # Check if adding the new sentence exceeds our target size limit
        if current_length + sentence_len > target_chunk_size and current_chunk:
            # Save the current sentence collection as a chunk
            chunks.append(" ".join(current_chunk))

            # Maintain sentence-level overlap
            if overlap_sentences > 0 and len(current_chunk) >= overlap_sentences:
                current_chunk = current_chunk[-overlap_sentences:]
                current_length = sum(len(s)
                                     for s in current_chunk) + len(current_chunk) - 1
            else:
                current_chunk = []
                current_length = 0

        current_chunk.append(sentence)
        current_length += sentence_len + 1

    # Append any remaining sentences in the final buffer
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# --- EXECUTION & SANITY CHECK ---
if __name__ == "__main__":
    file_path = "RAG-fromZero/data/paper.txt"

    print("--- STEP 1: Reading and Cleaning Data ---")
    cleaned_data = ingest_and_clean_text(file_path)
    print(
        f"File successfully loaded! Total Character Count: {len(cleaned_data)}")
    print(f"Raw Text Sample: \"{cleaned_data[:120]}...\"\n")

    print("--- STEP 2: Executing Sentence-Aware Chunking ---")
    chunks = sentence_aware_chunking(
        cleaned_data, target_chunk_size=500, overlap_sentences=1)
    print(f"Generated {len(chunks)} sentence-aware chunks.")

    print("\n--- CHUNK INSPECTION ---")
    print(f"[Chunk 1] Length: {len(chunks[0])} chars")
    print(f"Content: \"{chunks[0]}\"\n")

    if len(chunks) > 1:
        print(f"[Chunk 2] Length: {len(chunks[1])} chars")
        print(f"Content: \"{chunks[1]}\"\n")
