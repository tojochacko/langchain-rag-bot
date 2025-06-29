from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load env variables
load_dotenv()

# Load and split your document
loader = TextLoader("data.txt")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create vector store
embeddings = OpenAIEmbeddings()
vector_db = FAISS.from_documents(chunks, embeddings)

# Build retrieval chain
retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(OPENAPI_API_KEY), retriever=retriever)

# Ask a question
question = "What's this document about?"
response = qa_chain.run(question)
print("Answer:", response)
