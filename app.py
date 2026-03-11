import streamlit as st
import json
import pandas as pd
import os
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def load_tracks():
    with open('tracks.json', 'r') as file:
        return json.load(file)
# --- PAGE CONFIG ---
st.set_page_config(page_title="Faculty Advisor AI", layout="wide", page_icon="🎓")
st.title("🎓 Senpai - E-JUST Advisor")

# --- 1. DATA LOADING (Excel) ---
@st.cache_data
def load_professor_data():
    file_name = 'Professors_Data.xlsx'
    if not os.path.exists(file_name):
        return None
    try:
        return pd.read_excel(file_name, engine='openpyxl')
    except Exception:
        return None

# --- 2. PDF PROCESSING (Cloud-Friendly Embeddings) ---
@st.cache_resource
def process_pdf(file_path):
    if not os.path.exists(file_path):
        return None
    
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # HuggingFace is free and works in the cloud without Ollama
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db


profs_df = load_professor_data()
vdb = process_pdf("Full_HandBook.pdf")


def get_course_info(course_code):
    # Search logic to find the course inside your nested JSON
    schools = ejust_data["curriculum"]["PHASE_2_SCHOOLS"]
    for school in schools.values():
        for dept in school.get("departments", {}).values():
            for sem, courses in dept.get("semesters", {}).items():
                for c in courses:
                    if c["code"].upper() == course_code.upper():
                        return c
    return None


GROQ_API_KEY = "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"
client = Groq(api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Senpai ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        
        pdf_context = ""
        if vdb:
            docs = vdb.similarity_search(prompt, k=3)
            pdf_context = "\n".join([d.page_content for d in docs])

        
        prof_context = ""
        if profs_df is not None:
            query_lower = prompt.lower()
            for _, row in profs_df.iterrows():
                full_name = str(row.get('Name', '')).lower().replace('prof', '').replace('dr', '').strip()
                name_parts = [p for p in full_name.split() if len(p) > 2]
                
                if any(part in query_lower for part in name_parts):
                    
                    rating = row.get('Rating (1-5)') or row.get('Rating (1-5') or "N/A"
                    review = row.get('review_summary') or row.get('Review') or "No summary available."
                    
                    prof_context = (
                        f"Professor: {row.get('Name')}\n"
                        f"Rating: {rating}\n"
                        f"Review: {review}"
                    )
                    break

        
        system_prompt = f"""
        You are 'Senpai', a helpful academic advisor for E-JUST University.
        Use the following information to answer. If you don't know, say you aren't sure.
        
        Handbook Context: {pdf_context}
        Professor Context: {prof_context}
        
        Note: Students with CGPA < 2.0 are on academic probation and restricted to Half-Load.
        """

        #
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.2 # Lower temperature for better accuracy
            )
            ans = chat_completion.choices[0].message.content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Groq Error: {e}")