import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer


# 1. INITIALIZE SYSTEM COMPONENTS
print("Loading text embedding model...")
embedding_model = SentenceTransformer(
    "/home/ibrahim/works/nlp-llm-works/RAG-fromZero/model_files")


def read_plain_text_file(file_path):
    """
    Reads a raw text file using UTF-8 encoding.
    No complex parsers needed, just pure raw string extraction.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def split_text_into_chunks(text, chunk_size=600, overlap=60):
    """
    Slices the raw string linearly based on character index.
    The overlap ensures sentence boundaries aren't completely lost.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += (chunk_size - overlap)
    return [c for c in chunks if c]


# --- RUNNING THE PIPELINE ---

# Step 1: Read the text you copied from the internet
txt_path = "/home/ibrahim/works/nlp-llm-works/RAG-fromZero/data/paper.txt"
raw_text_content = read_plain_text_file(txt_path)
text_chunks = split_text_into_chunks(raw_text_content)

print(
    f"Successfully loaded text file. Generated {len(text_chunks)} manual chunks.")

# Step 2: Convert text chunks into numbers
print("Generating vectors for text chunks...")
chunk_embeddings = embedding_model.encode(text_chunks)

# --- STEP 2.5: PIPELINE VERIFICATION (SANITY CHECK) ---
print("\n" + "="*40)
print("     EMBEDDING PIPELINE VERIFICATION     ")
print("="*40)

# 1. Verify Raw Data Loading
total_chars = len(raw_text_content)
print(f"[1] Raw Text Analysis:")
print(f"    - Total character count: {total_chars}")
if total_chars == 0:
    print("    - ERROR: The file is empty or was not read correctly!")
else:
    print(f"    - Preview of source text (First 100 chars):")
    print(f"      \"{raw_text_content[:100].strip()}...\"")

# 2. Verify Chunking Logic & Content Continuity
total_chunks = len(text_chunks)
print(f"\n[2] Chunking Analysis:")
print(f"    - Total manual chunks generated: {total_chunks}")

if total_chunks > 0:
    print(f"    - First Chunk character length: {len(text_chunks[0])}")
    print(f"    - First Chunk Preview:")
    print(f"      \"{text_chunks[0][:150]}...\"")

    # Check if the overlap mechanism is functioning visually
    if total_chunks > 1:
        print(f"    - Second Chunk Preview (Look for the overlap at the start):")
        print(f"      \"{text_chunks[1][:150]}...\"")
else:
    print("    - ERROR: No chunks were generated. Check your split logic.")

# 3. Verify Matrix Shapes and Mathematical Integrity
print(f"\n[3] Embedding Matrix Analysis:")
print(f"    - Datatype of embeddings: {type(chunk_embeddings)}")

# Check numpy array dimensions
matrix_shape = chunk_embeddings.shape
print(f"    - Matrix Shape (Total Chunks, Vector Dimensions): {matrix_shape}")

# intfloat/multilingual-e5-small outputs exactly 384 dimensions
expected_dimension = 384
if matrix_shape[1] == expected_dimension:
    print(
        f"    - SUCCESS: Vector dimension matches the expected space ({expected_dimension}).")
else:
    print(
        f"    - WARNING: Dimension mismatch! Expected {expected_dimension}, got {matrix_shape[1]}")

# Safety check for numerical computing: Ensure no NaN (Not a Number) or Inf values exist
has_nan = np.isnan(chunk_embeddings).any()
has_inf = np.isinf(chunk_embeddings).any()
print(f"    - Contains NaN anomalies: {has_nan}")
print(f"    - Contains Infinite anomalies: {has_inf}")

print("="*40 + "\n")
