Angel One Support Chatbot
=========================

Project Overview
----------------

This project implements a Retrieval-Augmented Generation (RAG) chatbot that assists users by answering queries about Angel One and insurance options. The chatbot is trained on Angel One's support documentation and specific insurance PDFs to provide accurate and relevant information.

Features
--------

-   Web scraping of Angel One support documentation
-   Document processing for PDFs and DOCX files
-   Vectorized document storage using Pinecone
-   Semantic search for query matching
-   User-friendly Streamlit interface
-   Contextually relevant response generation

Architecture
------------

The system consists of four main components:

1.  **Web Scraper**: Collects and preprocesses text from Angel One's support website
2.  **Document Processor**: Converts PDFs and DOCX files into usable text chunks
3.  **Ingestion Pipeline**: Generates embeddings and uploads vectors to Pinecone
4.  **Chatbot Interface**: Streamlit-based UI for user interaction

Installation
------------

### Prerequisites

-   Python 3.8+
-   Pinecone API key
-   Required Python libraries (specified in requirements.txt)

### Setup

1.  Clone the repository:

```
git clone https://github.com/your-username/angel-one-support-chatbot.git
cd angel-one-support-chatbot

```

1.  Install dependencies:

```
pip install -r requirements.txt

```

1.  Create a `.env` file in the project root with your Pinecone credentials:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

```

1.  Prepare the data:
    -   Create a `data` folder and add your PDF and DOCX files
    -   Run the web scraper to gather Angel One support content

Usage
-----

### Data Preparation

1.  Run the web scraper to collect support documentation:

```
python web_scraper.py

```

1.  Process the documents:

```
python document_processor.py

```

1.  Ingest documents into Pinecone:

```
python ingestion.py

```

### Running the Chatbot

Launch the Streamlit application:

```
streamlit run chatbot_rag.py

```

The application will be available at http://localhost:8501
