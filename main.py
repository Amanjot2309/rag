from dotenv import load_dotenv
load_dotenv()

# from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain


def ingest_and_create_vectorstore():
    print("Loading and splitting documents...")
    # loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/v0.1")
    loader = TextLoader("data_entity.txt")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = splitter.split_documents(raw_documents)

    # for doc in documents:
    #     # Fix URLs in metadata
    #     new_url = doc.metadata["source"].replace("langchain-docs", "https:/")
    #     doc.metadata.update({"source": new_url})

    embeddings = OllamaEmbeddings(model="llama3")
    vectorstore = InMemoryVectorStore.from_documents(documents, embedding=embeddings)

    print(f"Ingested {len(documents)} chunks into vectorstore.")
    return vectorstore


def build_qa_chain(vectorstore):
    llm = OllamaLLM(model="llama3")

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


def main():
    vectorstore = ingest_and_create_vectorstore()
    qa_chain = build_qa_chain(vectorstore)

    # Ask a question here
    query = "How to make data entity?"
    print(f"\n🔍 Question: {query}")

    result = qa_chain.invoke({"input": query})
    print(f"\n📘 Answer: {result['answer']}")


if __name__ == "__main__":
    main()
