from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import faiss
import os

from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def get_embedding_model():
    """
    Get Ollama embedding model object.
    """
    return OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")

# def get_embeddings(model_emb, docs):
#     """
#     Split document into chunks and get embeddings from the param model object.
#     """
#     splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#     chunks = splitter.split_documents(docs)
#     embeddings = model_emb.embed_documents([chunk.page_content for chunk in chunks])

#     return embeddings

def load_document(file_paths: list[str]):
    """
    Load documents from multiple file paths. Supports PDF, CSV, and TXT files.
    File type is determined by the file extension.

    Args:
        file_paths: List of paths to document files (.pdf, .csv, or .txt)

    Returns:
        List of all loaded documents from all files

    Raises:
        ValueError: If any file type is not supported
    """
    all_docs = []

    for file_path in file_paths:
        # Get file extension
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        # Select appropriate loader based on file extension
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        elif ext == '.txt':
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported types: .pdf, .csv, .txt")

        # Load documents from this file and add to the collection
        docs = loader.load()
        all_docs.extend(docs)

    return all_docs

def get_vector_store(docs, index_path: str):
    """
    Load an existing FAISS index if it exists; otherwise build and persist one.
    """
    ollama_emb = get_embedding_model()

    # Check if the index already exists
    if os.path.exists(index_path):
        return FAISS.load_local(
            index_path, 
            ollama_emb,
            allow_dangerous_deserialization=True  # 👈 acknowledge trusted pickle
            )
    return build_vector_store(docs, index_path, ollama_emb)

def build_vector_store(docs, index_path: str, model_emb):
    """
    Create a FAISS vector store from `chunks` and save it to `index_path`.
    Uses Ollama-served embeddings instead of OpenAI.
    """
    # embeddings = get_embeddings(model_emb, docs)

    vector_db = FAISS.from_documents(docs, model_emb)
    vector_db.save_local(index_path)
    return vector_db

def main():
    # Load environment variables
    load_dotenv() # expects OPENAI_API_KEY in .env

    # List of files to process - can include PDF, CSV, or TXT files
    file_paths = ["data.txt"]          # Add more files as needed: ["data.txt", "document.pdf", "data.csv"]
    index_path = "vectors/faiss_index"      # Folder where the FAISS index is stored

    docs = load_document(file_paths)
    vector_db = get_vector_store(docs, index_path)

    retriever = vector_db.as_retriever()

    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
    )
    
    # Prepare to use the OpenAI model
    # llm = ChatOpenAI(
    #     model_name="gpt-3.5-turbo",
    #     temperature=0,
    #     openai_api_key=os.getenv("OPENAI_API_KEY"),  # ← read the env var
    # )

    # Define the prompt
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentence maximum and keep the answer concise. "
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, question_answer_chain)

    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        response = chain.invoke({"input": question})
        print("Answer:", response['answer'])

if __name__ == "__main__":
    main()

# Build retrieval chain
# retriever = vector_db.as_retriever()
# qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(OPENAPI_API_KEY), retriever=retriever)

# Ask a question
# question = "What's this document about?"
# response = qa_chain.run(question)
# print("Answer:", response)
