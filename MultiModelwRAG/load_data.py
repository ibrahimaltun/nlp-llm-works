import fitz
import chromadb
from sentence_transformers import SentenceTransformer
import os

# Enforce strict offline flags for Hugging Face components
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# prepare local db and embedding model, it full work offline
chroma_client = chromadb.PersistentClient(path="doc_db")
collection = chroma_client.get_or_create_collection(name="help_doc")
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# From folder, all files was downloaded manually and saved under the model_files folder
# https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main
embedding_model = SentenceTransformer("model_files")
print("Embedding model is loaded successfully.")

# images to be saved folder
svd_img_path = "saved_images"
os.makedirs(svd_img_path, exist_ok=True)

pdf_path = "optronics.pdf"
docc = fitz.open(pdf_path)

print("PDF is processing and saving to local db")

for page_nm in range(len(docc)):
    page = docc[page_nm]
    text = page.get_text()

    # extract images in the page
    image_list = page.get_images(full=True)
    saved_images_paths = []

    for img_idx, img in enumerate(image_list):
        xref = img[0]
        base_image = docc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]

        # save the image to disk
        img_name = f"page_{page_nm}_img_{img_idx}.{image_ext}"
        img_path = os.path.join(svd_img_path, img_name)
        with open(img_path, "wb") as ff:
            ff.write(image_bytes)
        saved_images_paths.append(img_path)

    # if page has texts, add it the db
    if text.strip():
        # text2vec
        vector = embedding_model.encode(text).tolist()

        metadata = {
            "page": page_nm + 1,
            "images": ",".join(saved_images_paths) if saved_images_paths else "nope"
        }

        collection.add(
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
            ids=[f"page_{page_nm}"]
        )

print("DB insertion is completed. System already know the doc")
