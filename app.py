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

def get_student_status(cgpa):
    if cgpa < 2.0:
        return "Half-Load (Academic Probation)", 14
    elif 2.0 <= cgpa < 3.0:
        return "Regular Load", 19
    else:
        return "Over-Achiever (Honors)", 21

def check_workload_safety(total_ch, limit):
    if total_ch > limit:
        return f"⚠️ **CRITICAL:** This schedule is {total_ch} CH, but your limit is {limit} CH. You MUST drop courses."
    return f"✅ This schedule is within your {limit} CH limit."
    
def get_priority_recommendation(sem_data, target_track=None):
    # Priorities: 
    # 1. Courses that are prerequisites for the student's desired track
    # 2. Hard core requirements
    # 3. Electives (Last priority)
    
    priority_list = []
    for c in sem_data:
        # Logic: If the course name matches the track keywords, move to top
        is_priority = False
        if target_track and target_track.lower() in c['name'].lower():
            is_priority = True
            
        priority_list.append({
            "code": c['code'],
            "name": c['name'],
            "ch": int(c.get('credit hours', 0)),
            "priority": is_priority
        })
    
    # Sort so priority courses are at the top
    return sorted(priority_list, key=lambda x: x['priority'], reverse=True)    

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
        sem_match = re.search(r"(?:semester|sem)\s*(\d)", prompt.lower())
        
        # Track detection (checks if they mentioned a specific field)
        track_keywords = ["power", "communications", "data", "software", "robotics", "embedded"]
        target_track = next((k for k in track_keywords if k in prompt.lower()), None)
        gpa_match = re.search(r"gpa\s*[:=]?\s*(\d(?:\.\d+)?)", prompt.lower())
        if gpa_match:
            user_cgpa = float(gpa_match.group(1))
        else:
            user_cgpa = 3.0 # Default if they don't say it
     

        if sem_match:
            sem_num = sem_match.group(1)
            summary, total_h = get_semester_summary(user_school, user_dept, f"semester_{sem_num}")
            if summary:
                safety_msg = check_workload_safety(total_h, max_credits)
                json_context = f"\n[ADVISOR DATA]:\nStatus: {status_name}\nMax Allowed: {max_credits} CH\n{safety_msg}\n{summary}"
                
                # If they are Half-Load and haven't picked a track, Senpai will prompt them
                if user_cgpa < 2.0 and not target_track:
                    json_context += "\n\nINSTRUCTION: The student is on Half-Load. Ask them which track they want to pursue so you can prioritize their 14 credits."
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
        Your student's CGPA is {user_cgpa}. 
        Their status is {status_name} with a limit of {max_credits} credits.
        
        Note: GPA < 2.0 = half load (Max 14 CH).
              2.0 < GPA < 3.0 = Regular load (Max 19 CH).
              GPA > 3.0 = over achiever (Max 21 CH).
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
