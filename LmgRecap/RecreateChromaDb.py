import chromadb, openai
from chromadb import Documents, EmbeddingFunction, Embeddings

class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        openai.api_key = "MYKEY"
        openai.api_base = "http://192.168.1.229:8081/v1"
        self._client = openai.Embedding

    def __call__(self, texts: Documents) -> Embeddings:
        texts = [t.replace("\n", " ") for t in texts]
        embeddings = self._client.create(input=texts, model="ggml-sfr-embedding-mistral-q8_0")["data"]
        sorted_embeddings = sorted(embeddings, key=lambda e: e["index"])
        return [result["embedding"] for result in sorted_embeddings]

client = chromadb.HttpClient(host='localhost', port=8000)
collection = client.delete_collection(name="Glossary")
collection = client.create_collection(name="Glossary", metadata={"hnsw:space": "cosine"}, embedding_function=CustomEmbeddingFunction())
