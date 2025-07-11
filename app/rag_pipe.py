import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA

load_dotenv()

llm = ChatOpenAI(model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()

def load_and_split_document(file_path):
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chanksphilly = splitter.split_documents(documents)
        return chanksphilly
    

    
    except Exception as e:
        print("Error: Could not load or split the document.")
        print(f"Details: {e}")
        raise RuntimeError(f"Error loading {file_path}") from e
    


def embed_and_store(docs):
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory="app/chroma_db")
    vectordb.persist()
    return vectordb

def initialize_vectorstore(file_path):
    if os.path.exists("chroma_db/index"):
        return Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    docs = load_and_split_document(file_path)
    return embed_and_store(docs)

def get_answer(query, vectorstore):
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())
    return qa.run(query)
