from dotenv import load_dotenv
load_dotenv()

import pickle

from langchain.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain


def run_llm(query: str):
    # Load vectorstore from pickle
    with open("vectorstore.pkl", "rb") as f:
        docsearch = pickle.load(f)

    chat = OllamaLLM(model="llama3")

    template = """
    Answer any user questions based solely on the context below:
    
    <context>
    {context}
    </context>
    
    If the answer is not provided in the context, say "Answer not in context".
    
    Question:
    {input}
    """
    retrieval_qa_chat_prompt = PromptTemplate.from_template(template=template)

    stuff_documents_chain = create_stuff_documents_chain(
        chat, retrieval_qa_chat_prompt
    )

    qa = create_retrieval_chain(
        retriever=docsearch.as_retriever(), combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke(input={"input": query})
    return result


if __name__ == "__main__":
    res = run_llm(query="How to make Pizza?")
    print(res["answer"])