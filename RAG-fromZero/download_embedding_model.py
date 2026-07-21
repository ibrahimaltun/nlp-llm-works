from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

# Model to be saved dir
model_dir = "RAG-fromZero/model_files/multilingual-e5-small"

snapshot_download(
    repo_id="intfloat/multilingual-e5-small",  # model repo
    local_dir=model_dir,
    max_workers=4
)

try:
    embedding_model = SentenceTransformer(model_dir)  # test it
    print("Model Download and loaded")
except Exception:
    print("Problem")
