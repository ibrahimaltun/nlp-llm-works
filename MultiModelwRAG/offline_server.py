import os
import base64
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import httpx

# Enforce strict offline flags for Hugging Face components
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

app = FastAPI(title="Qwen 2.5 VL RAG Gateway")

# Enable CORS for React UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading local offline system components...")

LOCAL_EMBEDDING_PATH = "model_files"
CHROMA_DB_PATH = "doc_db"

LLAMA_SERVE_URL = "http://127.0.0.1:8091"

print("Initializing local Vector DB and Embedding components...")
embedding_model = SentenceTransformer(LOCAL_EMBEDDING_PATH)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name="help_doc")
print("Orchestration pipeline ready.")


class QueryRequest(BaseModel):
    question: str


# def encode_image_to_base64(image_path: str) -> str:
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')


@app.post("/api/ask")
async def ask_question(request: QueryRequest):
    try:
        # Convert user inquiry into a vector embedding and query the local database
        query_vector = embedding_model.encode(request.question).tolist()
        query_results = collection.query(
            query_embeddings=[query_vector], n_results=1)

        # Guard clause if no matching reference text is found inside the database
        if (not query_results.get("documents") or
            len(query_results["documents"]) == 0 or
                len(query_results["documents"][0]) == 0):
            return {
                "response": "Kılavuzda bu konuyla ilgili herhangi bir eşleşme bulunamadı.",
                "source_page": None,
                "shown_image": None
            }

        # Extract retrieved text block and associated metadata safely
        retrieved_text = query_results["documents"][0][0] if query_results["documents"] else ""
        metadata = query_results["metadatas"][0][0] if query_results["metadatas"] else {
        }

        # associated_images = metadata.get(
        #     "images", "nope") if isinstance(metadata, dict) else "nope"
        source_page = metadata.get("page", None) if isinstance(
            metadata, dict) else None

        # Set up system instructions enforcing strict alignment with the manual
        # system_prompt = (
        #     "You are an expert assistant specialized in explaining the provided technical help documentation. "
        #     "Analyze the given source text and the associated visual image carefully to answer the user's question step-by-step. "
        #     "Respond clearly in Turkish. "
        #     "If the answer cannot be found in the provided context, do not make up facts. Politely state: 'Kılavuzda bu bilgiye ulaşamadım.'"
        # )
        system_prompt = (
            "You are an expert assistant specialized in explaining the provided technical help documentation. "
            "Analyze the given source text carefully to answer the user's question step-by-step. "
            "Respond clearly in Turkish. "
            "If the answer cannot be found in the provided context, do not make up facts. Politely state: 'Kılavuzda bu bilgiye ulaşamadım.'"
        )

        llama_payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Documentation Reference Context:\n{retrieved_text}\n\nUser Question: {request.question}"
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        # 5. Non-blocking asynchronous relay to C++ llama-serve process
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(LLAMA_SERVE_URL, json=llama_payload)

            if response.status_code != 200:
                raise HTTPException(
                    status_code=500,
                    detail=f"Local LLM service error: {response.text}"
                )

            completion_data = response.json()

            # Gelen cevabın güvenli şekilde doğrulanması ve ayıklanması
            if "choices" in completion_data and len(completion_data["choices"]) > 0:
                # [0] indeksini ekleyerek choices listesinin ilk elemanına erişiyoruz
                generated_answer = completion_data["choices"][0]["message"]["content"]
            else:
                raise HTTPException(
                    status_code=500, detail="Incomplete payload response from engine.")

        return {
            "response": generated_answer,
            "source_page": source_page,
            # "shown_image": frontend_image_payload
            "shown_image": None
        }

    except Exception as error_details:
        import traceback
        traceback.print_exc()  # <-- Prints the exact line number that failed into your console
        raise HTTPException(status_code=500, detail=str(error_details))


@app.get("/", response_class=HTMLResponse)
async def read_root():
    model_status_html = ""

    try:
        # Send a tiny, minimal payload to verify the C++ server connection
        test_payload = {
            "model": "local-model",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1  # Fast execution footprint
        }

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.post("http://127.0.0.1:8091/health", json=test_payload)

            if response.status_code == 200:
                model_status_html = """
                    <div class="badge success">AI Motoru Bağlantısı: AKTİF (Port 8091)</div>
                    <p style="color: #059669; font-weight: 500; font-size: 13px; margin-top: -5px;">
                        ✓ Qwen 2.5 VL modeli hazır ve istekleri işlemeye başladı.
                    </p>
                """
            else:
                raise Exception(f"Status code: {response.status_code}")

    except Exception as e:
        model_status_html = f"""
            <div class="badge error">AI Motoru Bağlantısı: KOPUK</div>
            <p style="color: #dc2626; font-weight: 500; font-size: 13px; margin-top: -5px;">
                ⚠️ llama-serve ayakta değil veya yanlış portta çalışıyor! (Hata: {str(e)})
            </p>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head>
        <meta charset="UTF-8">
        <title>Python API Gateway Status</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; text-align: center; padding-top: 8%; }}
            .card {{ background: white; max-width: 550px; margin: 0 auto; padding: 35px; border-radius: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; }}
            h1 {{ color: #0f172a; font-size: 24px; margin-bottom: 5px; font-weight: 700; }}
            .sub {{ font-size: 14px; color: #64748b; line-height: 1.6; margin-bottom: 25px; }}
            .badge {{ display: inline-block; padding: 6px 14px; font-weight: 700; border-radius: 30px; font-size: 11px; text-transform: uppercase; margin-bottom: 12px; tracking-spacing: 0.5px; }}
            .success {{ background: #def7ec; color: #03543f; }}
            .error {{ background: #fde8e8; color: #9b1c1c; }}
            .gateway-badge {{ background: #e0f2fe; color: #0369a1; font-weight: bold; padding: 4px 12px; border-radius: 6px; font-size: 12px; }}
            hr {{ border: 0; border-top: 1px solid #f1f5f9; my: 20px; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="card">
            <span class="gateway-badge">FastAPI Gateway: Port 8777</span>
            <h1 style="margin-top: 15px;">Python RAG Dağıtım Paneli</h1>
            <p class="sub">Bu katman ChromaDB vektör aramalarını yönetir ve verileri yapay zeka modeline iletir.</p>
            
            <hr />
            
            {model_status_html}
            
            <hr />
            
            <p style="font-size: 12px; color: #94a3b8;">Sistemi kullanmak için React uygulamanızı başlatın ve sorularınızı oradan sorun.</p>
        </div>
    </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8777)
