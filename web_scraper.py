import requests
from bs4 import BeautifulSoup
from readability import Document
import time
import os
import re

def fetch_preprocess_text(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}     # common convention
        res = requests.get(url, timeout=10, headers=headers)
        
        if res.status_code != 200:
            return None
        
        doc = Document(res.text)    # clean up text using readability -> Document
        summary_html = doc.summary()

        soup = BeautifulSoup(summary_html, "html.parser")

        text = soup.get_text(separator="\n", strip=True)
        
        # Clean up text using regex
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        
        text = re.sub(r'\.', '.\n', text)
        
        return text
    
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def scrape_pages():
    base_url = "https://www.angelone.in/support"    # error scraping 3-4 websites
    os.makedirs("documents", exist_ok=True)

    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")


    links = {a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith('http')}
    text_content = []

    for url in links:
        time.sleep(1)   # reduce load on server
        print(f"Scraping: {url}")
        text = fetch_preprocess_text(url)
        
        if text is not None:
            text_content.append(text)

    # write in file
    with open("documents/angelone_clean.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(text_content))

if __name__ == "__main__":
    scrape_pages()