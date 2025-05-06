# This will take in all websites, all pdfs and docx file content and convert it to
# vector database

from bs4 import BeautifulSoup 
import requests 
import re
import PyPDF2
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Create a new folder to save all text
output_folder = "extracted_text"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created new folder: {output_folder}")

# Websites
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
}

url = "https://www.angelone.in/support"

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, 'html.parser')

cat_grid = soup.find('div', class_='cat-grid')
grid_divs = cat_grid.find_all('div', class_='grid')

main_URLs = []
for grid in grid_divs:
    link = grid.find('a')['href']
    main_URLs.append(link)

websites = []
for main_url in main_URLs:
    try:
        # Make a request to the main URL
        response = requests.get(main_url, headers=headers)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the list-item div in this page
            list_item_div = soup.find('div', class_='list-item')
            
            # If found, extract all links from it
            if list_item_div:
                a_tags = list_item_div.find_all('a')
                
                # Get the href from each link
                sub_urls = [a['href'] for a in a_tags if 'href' in a.attrs]
                
                print(f"Found {len(sub_urls)} sub-URLs from {main_url}")
                websites.extend(sub_urls)
            else:
                print(f"No list-item div found in {main_url}")
        else:
            print(f"Failed to access {main_url}, status code: {response.status_code}")
    except Exception as e:
        print(f"Error processing {main_url}: {e}")

print(f"Total sub-URLs collected: {len(websites)}")

# PDFs
pdfs = [os.path.join("data", file) for file in os.listdir("data") if file.lower().endswith('pdf')]

# Document
docs = [os.path.join("data", file) for file in os.listdir("data") if file.lower().endswith('docx')]

all_documents = []

# 1. Load websites and save their content
for i, website in enumerate(websites):
    try:
        # Get the content
        web_response = requests.get(website, headers=headers)
        web_soup = BeautifulSoup(web_response.text, 'html.parser')
        
        # Extract text content
        text_content = web_soup.get_text(separator='\n', strip=True)
        
        # Create a filename based on the URL
        # Use the last part of the URL as a filename, or a counter if that fails
        url_parts = website.split('/')
        if len(url_parts) > 0 and url_parts[-1]:
            filename = f"website_{i+1}_{url_parts[-1]}.txt"
        else:
            filename = f"website_{i+1}.txt"
        
        # Save to file
        filepath = os.path.join(output_folder, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"Saved content from {website} to {filepath}")
        
        # Also load for processing
        loader = WebBaseLoader(website)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded {website}")
    except Exception as e:
        print(f"Error processing website {website}: {e}")

# 2. Load PDFs and save their content
for pdf_path in pdfs:
    try:
        # Get the filename without the directory
        pdf_filename = os.path.basename(pdf_path)
        output_path = os.path.join(output_folder, f"{pdf_filename}.txt")
        
        # Extract text using PyPDF2
        text_content = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"Saved text from {pdf_path} to {output_path}")
        
        # Load for document processing
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded {pdf_path}")
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")

# 3. Load DOCX and save their content
for docx_path in docs:
    try:
        # Get the filename without the directory
        docx_filename = os.path.basename(docx_path)
        output_path = os.path.join(output_folder, f"{docx_filename}.txt")
        
        # Load the document and get text
        loader = Docx2txtLoader(docx_path)
        documents = loader.load()
        
        # Extract text from documents
        text_content = ""
        for doc in documents:
            text_content += doc.page_content + "\n\n"
        
        # Save the extracted text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_content)
        
        print(f"Saved text from {docx_path} to {output_path}")
        
        # Add to all documents
        all_documents.extend(documents)
        print(f"Loaded {docx_path}")
    except Exception as e:
        print(f"Error processing DOCX {docx_path}: {e}")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

split_documents = text_splitter.split_documents(all_documents)
print(f"Split into {len(split_documents)} chunks")

# Save all_documents for later use
documents_path = os.path.join(output_folder, "all_documents.pkl")
with open(documents_path, 'wb') as f:
    pickle.dump(all_documents, f)

split_documents_path = os.path.join(output_folder, "split_documents.pkl")
with open(split_documents_path, 'wb') as f:
    pickle.dump(split_documents, f)

print(f"Saved all_documents and split_documents to {output_folder}")