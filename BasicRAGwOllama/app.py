import chainlit as cl
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# define local model
# Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)

# for lowest ram usage
Settings.llm = Ollama(model="qwen2.5:3b", request_timeout=120.0)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")


@cl.on_chat_start
async def start_chat():

    msg = cl.Message(
        content="Lokal yapay zeka başlatılıyor ve PDF analiz ediliyor. Lütfen bekleyin...")
    await msg.send()

    try:
        # DÜZELTİLEN KISIM: Okuyucuyu tanımla ve veri yüklemeyi asenkron çalıştır
        reader = SimpleDirectoryReader("./data")
        documents = await cl.make_async(reader.load_data)()

        # Lokalde vektör indeksini oluştur
        index = VectorStoreIndex.from_documents(documents)

        # Soru-Cevap motorunu hazırla
        query_engine = index.as_query_engine(streaming=True)

        cl.user_session.set("query_engine", query_engine)

        msg.content = "✅ Sistem tamamen offline ve ücretsiz olarak hazır! Sorularınızı sorabilirsiniz."
        await msg.update()

    except Exception as e:
        msg.content = f"Bir hata oluştu. Hata detayları: {str(e)}"
        await msg.update()


@cl.on_message
async def main(message: cl.Message):
    query_engine = cl.user_session.get("query_engine")

    # Kullanıcının sorusunu lokal modele gönder
    response = await cl.make_async(query_engine.query)(message.content)

    # Yanıtı ekrana akıt (streaming)
    msg = cl.Message(content="")
    for token in response.response_gen:
        await msg.stream_token(token)

    await msg.send()
