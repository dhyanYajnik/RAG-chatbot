import os
import pickle
from pinecone import Pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()

# Get API keys and index name from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Load the split documents
def load_documents(file_path="extracted_text/split_documents.pkl"):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Check if the index exists
    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"Warning: Index '{PINECONE_INDEX_NAME}' not found. Please create it first.")
    else:
        print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
    
    return pc.Index(name=PINECONE_INDEX_NAME)

# Load the embedding model
def load_embedding_model(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

# Process documents and upload to Pinecone
def process_and_upload_to_pinecone(documents, embedding_model, pinecone_index, batch_size=100):
    print(f"Processing {len(documents)} documents and uploading to Pinecone...")
    
    for i in tqdm(range(0, len(documents), batch_size)):
        # Get a batch of documents
        batch = documents[i:i+batch_size]
        
        # Prepare the batch data
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        
        # Generate embeddings
        embeddings = embedding_model.encode(texts)
        
        # Prepare vectors for Pinecone (new format for pinecone package)
        vectors = []
        for id, embedding, metadata in zip(ids, embeddings, metadatas):
            vectors.append((id, embedding.tolist(), metadata))
        
        # Upsert to Pinecone
        pinecone_index.upsert(vectors=vectors)
    
    print(f"Successfully uploaded {len(documents)} documents to Pinecone!")

# Main function
def main():
    print("Loading documents...")
    documents = load_documents()
    
    print("Initializing Pinecone...")
    pinecone_index = init_pinecone()
    
    print("Loading embedding model...")
    embedding_model = load_embedding_model()
    
    print("Processing documents and uploading to Pinecone...")
    process_and_upload_to_pinecone(documents, embedding_model, pinecone_index)
    
    print("Done! Vector database created successfully.")

if __name__ == "__main__":
    main()