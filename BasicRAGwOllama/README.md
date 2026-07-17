# RAG using PDF

## 1. Download LLM Model and Embedding Model

1. $ ollama pull llama3.2
2. $ ollama pull nomic-embed-text

## 2. Install pip packages

1. $ pip install llama-index llama-index-llms-ollama llama-index-embeddings-ollama pymupdf chainlit

## 3. Start the project

1. $ chainlit run app.py -w --port 8080

----

## Solve some issues

### 1. update python version to 3.10

1. $ conda activate env_name
2. $ conda install python=3.10
3. $ python --version

NOTE: repeat pip packages installation

----

### 2. ERROR: [Errno 98] error while attempting to bind on address ('127.0.0.1', 8000): address already in use

$ fuser -k 8000/tcp
$ kill -9 $(lsof -t -i:8000)
