import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = './data'
CHROMA_PATH = os.getenv('CHROMA_PATH', './chroma_db')
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 500))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 50))

embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

def ingest_all():
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.pdf'):
            path = os.path.join(DATA_DIR, fname)
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
            print(f'Loaded: {fname}')

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f'Total chunks created: {len(chunks)}')

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    print('Ingestion complete. ChromaDB is ready.')

if __name__ == '__main__':
    ingest_all()