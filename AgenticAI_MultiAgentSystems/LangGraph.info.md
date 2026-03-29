# AgenticAI System 
## LangGraph, Tool Calling vb. Notları

- tool calling için LangChain yerine LangGraph kullanmak artık daha kolay esnek ve modern. LangChain'de AgentExecutor yapısının karmaşık olması nedeniyle artık geride bırakıldı.

- LangChain, agents için tamamen LangGraph kullanımına odaklanmıştır.

- Local model kullanımı için ollama ya da vLLM kullanılabilir. biz ollama ile Llama modeli kullanıyoruz. İstenmesi halinde diğer modellerin entegrasyonu için <i>langchain-openai</i> paketi yüklenebilir.

- 

## environment kurulması ve paketlerin yüklenmesi
- $ conda create --name agentic_ai python=3.13
- $ pip install -U langchain langchain_ollama langraph
- $ pip install -U langchain-community langchain-core

## Ollama ile Model Kurulması
- $ ollama pull llama3.1
- $ ollama pull llama3-groq-tool-use
