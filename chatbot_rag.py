import os
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import re
from ingestion import embedding_model, pc, index


def retrieve_answer(query, k=5, score_threshold=0.4):
    query_vector = embedding_model.encode(query).tolist()
    result = index.query(vector=query_vector, top_k=k, include_metadata=True)
    
    matches = result.get("matches", [])
    
    if not matches:
        return "I don't know."
    
    print(f"Top match score: {matches[0].get('score', 0)}")
    
    relevant_texts = []
    for match in matches:
        if match.get("score", 0) >= score_threshold:
            text = match["metadata"].get("text", "")

            if text not in relevant_texts:
                relevant_texts.append(text)
    
    if not relevant_texts:
        return "I don't know."
        
    all_text = " ".join(relevant_texts)

    sentences = re.split(r'[.!?]+', all_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    # keyword based relevance scoring
    keywords = [w.lower() for w in query.split() if len(w) > 3]
    
    scored_sentences = []
    for sentence in sentences:
        score = sum(1 for keyword in keywords if keyword in sentence.lower())
        scored_sentences.append((sentence, score))
    
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)
    
    top_sentences = [s[0] for s in sorted_sentences[:3]]
    
    # fallback when no keyword matched
    if not top_sentences:
        return relevant_texts[0][:300]
        
    answer = ". ".join(top_sentences)
    if not answer.endswith("."):
        answer += "."
        
    return answer

# UI
st.title("Angel One Support Chatbot")

user_input = st.text_input("Ask a question about Angel One or insurance options:")

if user_input:
    with st.spinner("Searching..."):
        answer = retrieve_answer(user_input)
        
    st.write(f"**User Query:** {user_input}")
    st.write(f"**Answer:** {answer}")