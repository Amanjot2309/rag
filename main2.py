import os
import time
import jwt
import requests
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from github import Github
# from github_webhook import create_app
# from github_webhook import WebhookCommonPayload
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

# FastAPI + Webhook App setup
app = FastAPI()
# hook_app = create_app(secret_token=os.getenv("GITHUB_WEBHOOK_SECRET"))
# app.mount("/", hook_app)  # Mount webhook router

### LangChain QA Setup ###

class QueryRequest(BaseModel):
    question: str

vectorstore = None
qa_chain = None

def ingest_and_create_vectorstore():
    loader = TextLoader("pizza_recipe.txt")
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = splitter.split_documents(raw_documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vs = InMemoryVectorStore.from_documents(documents, embedding=embeddings)
    print(f"Ingested {len(documents)} chunks into vectorstore.")
    return vs

def build_qa_chain(vectorstore):
    llm = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")
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
    print("Application startup complete.")

@app.post("/query")
def query_qa(request: QueryRequest):
    result = qa_chain.invoke({"input": request.question})
    return {"answer": result["answer"]}

### GitHub App Auth Endpoint ###

@app.get("/github-app/callback")
def github_app_callback(installation_id: int):
    private_key = open(os.getenv("GITHUB_APP_PRIVATE_KEY_PATH")).read()
    app_id = os.getenv("GITHUB_APP_ID")

    jwt_token = jwt.encode(
        {"iat": int(time.time()), "exp": int(time.time()) + 600, "iss": app_id},
        private_key, algorithm="RS256"
    )

    res = requests.post(
        f"https://api.github.com/app/installations/{installation_id}/access_tokens",
        headers={"Authorization": f"Bearer {jwt_token}", "Accept": "application/vnd.github+json"}
    )
    res.raise_for_status()
    token = res.json()["token"]

    gh = Github(token)
    return {"message": "Authenticated as GitHub App installation."}

### Webhook Handler ###

# class PRPayload(WebhookCommonPayload):
#     action: str
#     pull_request: dict

# @hook_app.hooks.register("pull_request", PRPayload)
# async def pr_handler(payload: PRPayload):
#     print(f"PR event: {payload.action} in {payload.repository.full_name}")
#     # Add logic: comment, QA triggered, etc.
