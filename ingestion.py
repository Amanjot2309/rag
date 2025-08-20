from dotenv import load_dotenv

load_dotenv()

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
# from langchain_pinecone import PineconeVectorStore

import pickle

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = OllamaEmbeddings(model="llama3")

def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/v0.1")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
    print("****Loading to vectorstore done ***")

    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    print("Vectorstore saved to vectorstore.pkl")

if __name__ == "__main__":
    ingest_docs()
