# app.py
import streamlit as st
from rag import RAGSystem
import time

# Page configuration
st.set_page_config(
    page_title="Document Q&A System",
    page_icon="🔍",
    layout="wide"
)

# App title and description
st.title("📚 Document Q&A System")
st.markdown("""
This system uses RAG (Retrieval-Augmented Generation) to answer questions based on your documents.
Ask a question, and the system will search for relevant information and generate a response.
""")

# Initialize the RAG system with caching to avoid reloading on each query
@st.cache_resource
def initialize_rag():
    with st.spinner("Initializing RAG system (this may take a moment)..."):
        return RAGSystem()

# Create session state variables if they don't exist
if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = []

if "source_info" not in st.session_state:
    st.session_state.source_info = []

if "answer" not in st.session_state:
    st.session_state.answer = ""

if "query_time" not in st.session_state:
    st.session_state.query_time = 0

# Sidebar with settings
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=10, value=3)
    
    st.header("About")
    st.info("""
    This application uses:
    - Pinecone for vector storage
    - Sentence Transformers for embeddings
    - TinyLlama for response generation
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Ask a Question")
    
    # Question input
    question = st.text_input("Your question:", 
                           placeholder="e.g., How do I open a demat account?")

    # Submit button
    submit = st.button("Get Answer", type="primary")
    
    # Process question when submitted
    if submit and question:
        # Clear previous results
        st.session_state.retrieved_docs = []
        st.session_state.source_info = []
        st.session_state.answer = ""
        
        # Initialize RAG system
        rag = initialize_rag()
        
        # Start timer
        start_time = time.time()
        
        # Retrieve documents
        with st.spinner("Searching for relevant documents..."):
            documents, source_info = rag.retrieve_documents(question, top_k=top_k)
            st.session_state.retrieved_docs = documents
            st.session_state.source_info = source_info
        
        # Check if documents were found
        if not st.session_state.retrieved_docs:
            st.error("No relevant documents found. Please try a different question.")
            st.session_state.query_time = time.time() - start_time
        else:
            # Generate answer
            with st.spinner("Generating answer..."):
                st.session_state.answer = rag.generate_response(question, documents, source_info)
            
            # Calculate total query time
            st.session_state.query_time = time.time() - start_time

with col2:
    st.subheader("Answer")
    
    # Display the answer if available
    if st.session_state.answer:
        st.markdown(f"**Question:** {question}")
        st.markdown(st.session_state.answer)
        st.caption(f"Query processed in {st.session_state.query_time:.2f} seconds")
        
    # Show documents if available
    if st.session_state.retrieved_docs:
        with st.expander("View Source Documents", expanded=False):
            for i, (doc, source) in enumerate(zip(st.session_state.retrieved_docs, 
                                              st.session_state.source_info)):
                # Display source information if available
                if source and 'source' in source:
                    st.markdown(f"**Document {i+1}** - Source: {source['source']} (Relevance: {source.get('score', 'N/A'):.2f})")
                else:
                    st.markdown(f"**Document {i+1}**")
                
                # Display document content
                st.text_area(f"Content", doc, height=150, key=f"doc_{i}")

# Display usage instructions at the bottom
st.markdown("---")
st.markdown("""
### Usage Tips:
1. Ask specific questions to get better answers
2. Adjust the number of documents in the sidebar if needed
3. Check the source documents if you want to verify the information
""")