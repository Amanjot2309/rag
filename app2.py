from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

# Global variables to hold vectorstore and QA chain
vectorstore = None
qa_chain = None

def ingest_and_create_vectorstore():
    print("Loading pizza recipe from text file...")
    loader = TextLoader("pizza_recipe.txt")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)

    print(f"Ingested {len(documents)} chunks into vectorstore.")
    return vectorstore


def build_qa_chain(vectorstore):
    pipe = pipeline("text-generation", model="distilbert/distilgpt2")
    llm = HuggingFacePipeline(pipeline=pipe)

    prompt_template = """
    Answer the user's question using **only** the context below.

    <context>
    {context}
    </context>

    If the answer isn't in the context, say "Answer not in context".

    Question:
    {input}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

@app.on_event("startup")
def startup_event():
    global vectorstore, qa_chain
    vectorstore = ingest_and_create_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

@app.post("/query")
def query_qa(request: QueryRequest):
    global qa_chain
    result = qa_chain.invoke({"input": request.question})
    return {"answer": result["answer"]}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app2:app", host="0.0.0.0", port=8000, reload=True)
