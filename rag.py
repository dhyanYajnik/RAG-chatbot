import os
import pickle
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load environment variables
load_dotenv()

# Get API keys and index name from environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Model parameters
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class RAGSystem:
    def __init__(self):
        """Initialize the RAG system with embedding model, LLM, and vector DB connection."""
        print("Initializing RAG system...")
        
        # Load embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Initialize Pinecone
        print("Connecting to Pinecone...")
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(PINECONE_INDEX_NAME)
        
        # Check index status
        try:
            stats = self.index.describe_index_stats()
            vector_count = stats.get('total_vector_count', 0)
            print(f"Connected to Pinecone index with {vector_count} vectors")
        except Exception as e:
            print(f"Warning: Error checking Pinecone stats: {e}")
        
        # Load language model for generation
        print(f"Loading language model: {LLM_MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_NAME,
            device_map="auto",
        )
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1,
            truncation=True
        )
        
        print("RAG system initialized and ready!")

    def get_embeddings(self, text):
        """Generate embeddings for the given text."""
        try:
            return self.embedding_model.encode(text)
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return empty embedding as fallback (this should be handled by calling code)
            return []

    def retrieve_documents(self, query, top_k=3):
        """Retrieve relevant documents from Pinecone using the query embeddings."""
        print(f"Retrieving documents for query: '{query}'")
        documents = []
        source_info = []
        
        try:
            # Generate query embedding
            query_embedding = self.get_embeddings(query)
            
            if len(query_embedding) == 0:
                print("Failed to generate query embedding")
                return documents, source_info
                
            print(f"Generated query embedding with dimension {len(query_embedding)}")
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            print(f"Retrieved {len(results.get('matches', []))} matches from Pinecone")
            
            # Process results
            for match in results.get('matches', []):
                score = match.get('score', 0)
                print(f"Match score: {score}")
                
                # Extract text from metadata
                if 'metadata' in match:
                    metadata = match['metadata']
                    
                    # Try to construct a useful document from available metadata
                    doc_parts = []
                    
                    # Add title if available
                    if 'title' in metadata and metadata['title']:
                        doc_parts.append(f"Title: {metadata['title']}")
                    
                    # Add description if available
                    if 'description' in metadata and metadata['description']:
                        doc_parts.append(f"Description: {metadata['description']}")
                    
                    # If neither title nor description, try other fields
                    if not doc_parts and 'text' in metadata:
                        doc_parts.append(metadata['text'])
                    elif not doc_parts and 'page_content' in metadata:
                        doc_parts.append(metadata['page_content'])
                    
                    # If we have any content, create a document
                    if doc_parts:
                        doc_text = "\n".join(doc_parts)
                        documents.append(doc_text)
                        
                        # Store source info for citation
                        source = metadata.get('source', 'Unknown source')
                        source_info.append({
                            'source': source,
                            'score': score
                        })
                    else:
                        # Log available metadata fields for debugging
                        print(f"Warning: No usable content found in metadata. Available fields: {list(metadata.keys())}")
            
            print(f"Successfully extracted {len(documents)} documents")
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
        
        return documents, source_info
    
    def generate_response(self, query, documents, source_info=None):
        """Generate a response using the LLM with retrieved documents as context."""
        print("Generating response...")
        
        if not documents:
            return "I couldn't find any relevant information to answer your question. Please try rephrasing your question or check that documents are properly indexed."
        
        try:
            # Prepare context from documents
            context_parts = []
            
            # Add each document with its source
            for i, (doc, source) in enumerate(zip(documents, source_info or [{}] * len(documents))):
                # Trim if too long
                max_doc_length = 500
                doc_text = doc[:max_doc_length] + "..." if len(doc) > max_doc_length else doc
                
                # Add source information if available
                source_text = f" (Source: {source.get('source', 'Unknown')})" if source else ""
                
                # Add to context
                context_parts.append(f"Document {i+1}{source_text}:\n{doc_text}")
            
            # Join all context parts
            context = "\n\n".join(context_parts)
            
            # Create prompt with context and query
            prompt = f"""<CONTEXT>
{context}
</CONTEXT>

Based on the context information provided above, please answer the following question:
{query}

Answer:"""
            
            # Generate response
            response = self.generator(prompt, max_length=512)[0]['generated_text']
            
            # Extract just the generated answer (after the prompt)
            answer = response[len(prompt):].strip()
            
            if not answer:
                return "I processed your query but couldn't generate a meaningful response. Please try rephrasing your question."
                
            return answer
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"An error occurred while generating a response. Error: {str(e)}"

    def query(self, question):
        """End-to-end RAG pipeline: retrieve documents and generate response."""
        print(f"\nProcessing query: {question}")
        
        # Step 1: Retrieve relevant documents
        print("Retrieving relevant documents...")
        documents, source_info = self.retrieve_documents(question)
        
        # Print document count
        print(f"Retrieved {len(documents)} documents")
        
        # Step 2: Generate response
        print("Generating response...")
        response = self.generate_response(question, documents, source_info)
        
        return response