import os
import time
import uuid
from dotenv import load_dotenv
from document_processor import process_documents, clean_text
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# text embedding using sentence transformer
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def get_embedding(text):
    return embedding_model.encode(text).tolist()

def ingest_documents():
    load_dotenv()

    # pinecone for indexing
    index_name = os.environ["PINECONE_INDEX_NAME"]
    pinecone_api_key = os.environ["PINECONE_API_KEY"]

    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in [index.name for index in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

        # wait for index to be ready
        while True:
            status = pc.describe_index(index_name).status["ready"]
            if status:
                break
            time.sleep(1)

    index = pc.Index(index_name)

    documents = process_documents()
    
    try:
        web_content_path = "documents/angelone_clean.txt"
        if os.path.exists(web_content_path):
            with open(web_content_path, "r", encoding="utf-8") as f:
                web_text = f.read()
            
            # web text to chunk
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=200,
                chunk_overlap=50,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            web_chunks = splitter.create_documents([web_text])
            
            for chunk in web_chunks:
                chunk.metadata["source"] = "angelone_website"
            
            documents.extend(web_chunks)

    except Exception as e:
        print(f"Error processing web content: {str(e)}")
    
    if not documents:
        return

    doc_ids = [str(uuid.uuid4()) for _ in documents]
    print(f"Generating embeddings for {len(documents)} documents")

    vectors = []
    for i, doc in enumerate(documents):
        embedding = get_embedding(doc.page_content)
        vectors.append((doc_ids[i], embedding, {"text": doc.page_content}))

    print("Uploading to Pinecone")
    for i in range(0, len(vectors), 100):
        batch = vectors[i:i + 100]
        index.upsert(vectors=batch)
        print(f"Uploaded batch {i//100 + 1}/{(len(vectors) + 99)//100}")
        time.sleep(1)

    print(f"Uploaded {len(documents)} documents to Pinecone")

if __name__ == "__main__":
    ingest_documents()