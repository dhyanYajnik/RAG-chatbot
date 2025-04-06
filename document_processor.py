import os
import re
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    text = text.replace('. ', '.\n')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+ of \d+', '', text)
    return text

def process_documents():
    data_folder = "data"
    output_folder = "documents"
    os.makedirs(output_folder, exist_ok=True)

    files = [
        "America's_Choice_2500_Gold_SOB (1) (1).pdf",
        "America's_Choice_5000_Bronze_SOB (2).pdf",
        "America's_Choice_5000_HSA_SOB (2).pdf",
        "America's_Choice_7350_Copper_SOB (1) (1).pdf",
        "America's_Choice_Medical_Questions_-_Modified_(3) (1).docx"
    ]

    documents = []

    for file in files:
        path = os.path.join(data_folder, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = Docx2txtLoader(path)

        try:
            docs = loader.load()

            for doc in docs:
                doc.page_content = clean_text(doc.page_content)

            documents.extend(docs)

        except Exception as e:
            print(f"Error loading {file}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    # split into chunks
    chunks = splitter.split_documents(documents)

    with open(os.path.join(output_folder, "all_text.txt"), "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.page_content.strip() + "\n\n")

    return chunks

if __name__ == "__main__":
    docs = process_documents()