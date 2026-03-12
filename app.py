import streamlit as st
import json
import pandas as pd
import os
import re
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- 1. DATA LOADING ---
def load_tracks():
    if os.path.exists('Tracks.json'):
        with open('Tracks.json', 'r') as file:
            return json.load(file)
    return None

ejust_data = load_tracks()

@st.cache_data
def load_professor_data():
    file_name = 'Professors_Data.xlsx'
    if not os.path.exists(file_name): return None
    try:
        return pd.read_excel(file_name, engine='openpyxl')
    except Exception: return None

@st.cache_resource
def process_pdf(file_path):
    if not os.path.exists(file_path): return None
    loader = PyPDFLoader(file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vector_db

profs_df = load_professor_data()
vdb = process_pdf("Full_HandBook.pdf")

# --- 2. SEMESTER SUMMARY ALGORITHM ---
def get_semester_summary(school_key, dept_key, sem_key):
    try:
        if not ejust_data: return "Database not found.", 0
        
        # Check if Foundation or Dept
        if sem_key in ["semester_1", "semester_2", "semester_3"]:
            sem_data = ejust_data["curriculum"]["PHASE_1_FOUNDATION"][sem_key]
            title_prefix = "Foundation"
        else:
            sem_data = ejust_data["curriculum"]["PHASE_2_SCHOOLS"][school_key]["departments"][dept_key]["semesters"][sem_key]
            title_prefix = dept_key

        course_details = []
        total_ch = 0
        for c in sem_data:
            name = c.get("name", "Unknown")
            code = c.get("code", "???")
            prereq = c.get("prereq") or "None"
            ch_raw = c.get("credit hours", 0)
            ch = int(ch_raw) if str(ch_raw).isdigit() else 0
            total_ch += ch
            course_details.append(f"| {code} | {name} | {prereq} | {ch} CH |")
        
        header = "| Code | Course Name | Prerequisite | Credits |\n| :--- | :--- | :--- | :--- |"
        table = "\n".join(course_details)
        summary = f"### 📚 {title_prefix} - {sem_key.replace('_', ' ').title()}\n{header}\n{table}\n\n**Total Workload: {total_ch} Credit Hours**"
        return summary, total_ch
    except Exception:
        return None, 0

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="Senpai - EJUST Advisor", layout="wide", page_icon="🎓")
st.title("🎓 Senpai - E-JUST Advisor")

# --- 4. CHAT LOGIC ---
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
        # A. PDF RAG
        pdf_context = ""
        if vdb:
            docs = vdb.similarity_search(prompt, k=3)
            pdf_context = "\n".join([d.page_content for d in docs])

        # B. AUTOMATIC JSON FETCHING (Detection Logic)
        json_context = ""
        # 1. Detect Semester Number
        sem_match = re.search(r"(?:semester|sem)\s*(\d)", prompt.lower())
        # 2. Detect Department (Defaulting to CSE if not mentioned)
        dept_match = re.search(r"(cse|ece|mte|cpe|eece)", prompt.lower())
        target_dept = dept_match.group(1).upper() if dept_match else "CSE"
        
        if sem_match:
            sem_num = sem_match.group(1)
            summary, _ = get_semester_summary("ECCE", target_dept, f"semester_{sem_num}")
            if summary:
                json_context = f"\n[OFFICIAL CURRICULUM]:\n{summary}"

        # C. Professor Logic
        prof_context = ""
        if profs_df is not None:
            q_low = prompt.lower()
            for _, row in profs_df.iterrows():
                p_name = str(row.get('Name', '')).lower()
                if any(part in q_low for part in p_name.split() if len(part) > 2):
                    prof_context = f"Professor: {row.get('Name')}\nRating: {row.get('Rating (1-5)', 'N/A')}\nReview: {row.get('Review', 'No summary')}"
                    break

        system_prompt = f"""
        You are 'Senpai', the E-JUST advisor. 
        Use the data provided to explain rules and courses.
        
        {json_context}
        
        Handbook: {pdf_context}
        Professor: {prof_context}
        
        Note: GPA < 2.0 = Probation (Max 14 CH).
        """

        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.2
            )
            ans = chat_completion.choices[0].message.content
            st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})
        except Exception as e:
            st.error(f"Groq Error: {e}")
