# AgenticAI System 
## LangGraph, Tool Calling vb. Notları

- tool calling için LangChain yerine LangGraph kullanmak artık daha kolay esnek ve modern. LangChain'de AgentExecutor yapısının karmaşık olması nedeniyle artık geride bırakıldı.

- LangChain, agents için tamamen LangGraph kullanımına odaklanmıştır.

- Local model kullanımı için ollama ya da vLLM kullanılabilir. biz ollama ile Llama modeli kullanıyoruz. İstenmesi halinde diğer modellerin entegrasyonu için <i>langchain-openai</i> paketi yüklenebilir.

- 

----

## environment kurulması ve paketlerin yüklenmesi
- $ conda create --name agentic_ai python=3.13
- $ pip install -U langchain langchain_ollama langraph
- $ pip install -U langchain-community langchain-core

## ollama ile Model Kurulması
Önce ollama kurulması gerekmektedir: https://ollama.com/download
- $ ollama pull llama3.1
- $ ollama pull llama3-groq-tool-use

----

### Çalıştırma Adımları
- tool olarak kullanmak istediğimiz metotlarımızı yazıyoruz ve bunların hepsini bir liste olarak tutuyoruz. örn: tools = []

- Ardından local modelimizi Ollama üzerinden okuma işlemi yapıyoruz. Burada kullandığımız modelin tool calling destekleyen bir model versiyonu olmasına dikkat ediyoruz. örn: llama3-ggroq-tool-use. Standart Llama3 modeli tool calling işlemlerinde bocalayabilir bu yüzden bu tarz bir model kullanmak başarı oranını arttırrır.

- model ile tool'lar bind_tools ile bağlanır, model burada kullanabileceği araçları bilir.

- graph state tanımlanması -> node'ların tanımlanması

- graph oluşturma(edges ve logic) -> workflow ayarlamaları ve derleme

- sorgu belirleme -> modele verme -> app.stream -> Yanıtı Gösterme
