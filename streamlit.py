import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

@st.cache_resource(show_spinner=True)
def ingest_and_create_vectorstore():
    st.write("Loading pizza recipe from text file...")
    loader = TextLoader("pizza_recipe.txt")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = splitter.split_documents(raw_documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)

    st.write(f"Ingested {len(documents)} chunks into vectorstore.")
    return vectorstore

@st.cache_resource(show_spinner=True)
def _build_qa_chain(_vectorstore):
    pipe = pipeline(
        "text-generation",
        model="distilbert/distilgpt2",
        device=-1  # Use CPU
    )
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
        retriever=_vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

def main():
    st.title("🍕 Pizza Recipe Q&A")

    vectorstore = ingest_and_create_vectorstore()
    qa_chain = _build_qa_chain(vectorstore)

    question = st.text_input("How to make pizza?")

    if question:
        with st.spinner("Generating answer..."):
            result = qa_chain.invoke({"input": question})
        st.markdown("### Answer:")
        st.write(result["answer"])

if __name__ == "__main__":
    main()
