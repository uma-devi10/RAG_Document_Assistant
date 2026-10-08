from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

def create_vector_store(text):

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_db = FAISS.from_texts(docs, embeddings)

    return vector_db