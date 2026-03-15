import streamlit as st
import json
import pandas as pd
import os
import re
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import time
import base64

def load_file_b64(path, mime):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return ""

font_b64 = load_file_b64("fonts/LEMONMILK-Bold.otf", "font/otf")
owl_b64  = load_file_b64("assets/owl.png", "image/png")
wave_b64 = load_file_b64("assets/wave.png", "image/png")
# ---------------------------
# CSS + font-face + layout
# ---------------------------
css = f"""
<style>

@font-face {{
    font-family: 'LemonMilk';
    src: url('{font_b64}') format('opentype');
    font-weight: 700;
    font-style: normal;
}}

/* Reset & Global */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"],
section[data-testid="stSidebar"],
.main {{
    background-color: #ffffff !important;
    color: #1a1a1a !important;
}}

* {{
    color: #1a1a1a !important;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

 /* Header */
.header-container {{
    display: flex;
    align-items: center;
    gap: 18px;
    padding: 28px 48px;
    position: relative;
    top: -100px;
    left: -300px;
    z-index: 3;
}}
.logo {{
    width: 60px;
    height: auto;
    display: block;
}}
.senpai-title {{
    font-family: 'LemonMilk', 'Arial Black', sans-serif;
    font-size: 50px;
    font-weight: 700;
    letter-spacing: 4px;
    color: 
#1a1a1a;
    line-height: 1;
}}

/* Wave decoration (top-right) */
.wave {{
    position: fixed;
    top: -300px;
    right: -320px;
    width: 1000px;
    z-index: 0;
    pointer-events: none;
    opacity: 0.95;
}}

/* Chat messages */
[data-testid="stChatMessage"] {{
    background: #ffffff !important;
    border-radius: 14px !important;
    padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06) !important;
    position: relative;
    z-index: 1;
    color: #1a1a1a !important;
}}

/* ── Input container — remove dark background ── */
[data-testid="stBottom"] {{
    background: linear-gradient(to top, #ffffff 80%, rgba(255,255,255,0)) !important;
    padding: 16px 60px 28px !important;
    border: none !important;
}}

[data-testid="stBottom"] > div {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* ── Input pill ── */
[data-testid="stChatInput"] {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}}

[data-testid="stChatInput"] textarea {{
    border-radius: 50px !important;
    border: 2px solid #d2d2d2 !important;
    padding: 16px 26px !important;
    font-size: 15px !important;
    background: #ffffff !important;
    box-shadow: 0 2px 20px rgba(0,0,0,0.08) !important;
    color: #1a1a1a !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}}

[data-testid="stChatInput"] textarea:focus {{
    border-color: #c8291a !important;
    box-shadow: 0 0 0 3px rgba(200,41,26,0.1) !important;
    outline: none !important;
}}

[data-testid="stChatInput"] textarea::placeholder {{
    color: #aaa !important;
    font-size: 15px !important;
}}

/* ── Send button — red circle ── */
[data-testid="stChatInput"] button {{
    background: #c8291a !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    border: none !important;
    color: white !important;
}}

[data-testid="stChatInput"] button:hover {{
    background: #a82215 !important;
    transform: scale(1.05) !important;
}}

[data-testid="stChatInput"] button svg {{
    fill: white !important;
    color: white !important;
}}

/* ── White gradient behind input so chat doesn't show through ── */
[data-testid="stBottom"] {{
    background: linear-gradient(to top, #ffffff 75%, rgba(255,255,255,0)) !important;
    padding: 16px 40px 28px !important;
}}

@media (max-width: 768px) {{
    .header-container {{
        padding: 1px 10px;
    }}
    .logo {{ width: 48px; }}
    .senpai-title {{ font-size: 28px; letter-spacing: 2px; }}
    .wave {{ width: 260px; }}
}}

/* Mobile adjustments */
@media (max-width: 768px) {{
  .header-container {{ padding: 12px 16px; }}
  .logo {{ width: 60px; position: relative; top: -120px; right: 0px;}}
  .senpai-title {{ font-size: 28px; position: relative; letter-spacing: 2px; top: 0px; right: 0px; }}
  .wave {{ width: 400px; top: -120px; right: -100px; }}
  [data-testid="stChatMessage"] {{ max-width: calc(100% - 28px) !important; margin-left: 0px !important; margin-right: 0px !important; }}
  [data-testid="stChatInput"] {{ max-width: 100% !important; }}
  [data-testid="stChatInput"] textarea {{ font-size: 16px !important; padding: 14px 16px !important; }}
}}

/* Small screens / very short viewports: reduce bottom padding */
@media (max-height: 600px) {{
  .main > div {{ padding-bottom: 120px !important; }}
  [data-testid="stChatInput"] textarea {{ padding: 12px 14px !important; }}
  }}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# show wave image with fallback if file missing
if wave_b64:
    st.markdown(f'<img src="{wave_b64}" class="wave">', unsafe_allow_html=True)

# Header: owl logo + title
logo_html = f"""
<div class="header-container">
    <img src="{owl_b64}" class="logo" alt="Senpai owl logo">
    <div class="senpai-title">SENPAI</div>
</div>
"""
st.markdown(logo_html, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ════════════════════════════════════════════════════════════════

def load_ejust_data():
    if os.path.exists('Tracks.json'):
        with open('Tracks.json', 'r') as f:
            return json.load(f)
    return None

ejust_data = load_ejust_data()

@st.cache_data
def load_professor_data():
    if not os.path.exists('Professors_Data.xlsx'):
        return None
    try:
        return pd.read_excel('Professors_Data.xlsx', engine='openpyxl')
    except Exception:
        return None

@st.cache_resource
def process_pdf(path):
    if not os.path.exists(path):
        return None
    loader   = PyPDFLoader(path)
    docs     = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks   = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma.from_documents(documents=chunks, embedding=embeddings)

profs_df = load_professor_data()
vdb      = process_pdf("Full_HandBook.pdf")

# ════════════════════════════════════════════════════════════════
# 2. CGPA → STATUS
# ── CHANGED: treat half-load as regular (per your request) ──
# ════════════════════════════════════════════════════════════════

def get_student_status(cgpa: float) -> tuple[str, int]:
    """
    Modified to make students Regular Load even if CGPA < 2.0,
    per the request. Returns (label, max_ch).
    """
    # NOTE: original policy was:
    #   <2.0 -> Half-Load (14 CH)
    #   2.0-2.99 -> Regular (19 CH)
    #   >=3.0 -> Over-Achiever (21 CH)
    # User asked to change current half-load status to Regular,
    # so below we map <2.0 to Regular Load (19 CH).
    if cgpa < 2.0:
        return "Regular Load", 19
    elif cgpa < 3.0:
        return "Regular Load", 19
    else:
        return "Over-Achiever (Honors)", 21

# ════════════════════════════════════════════════════════════════
# 3. TRACK MAP
# ════════════════════════════════════════════════════════════════

TRACK_MAP: dict[str, tuple[str, str, str]] = {
    "cse":           ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    "computer":      ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    "software":      ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    "ece":           ("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "electronics":   ("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "communications":("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "communication": ("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "epe":           ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    "power":         ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    "electrical":    ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    "mie":           ("ECCE", "MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    "biomedical":    ("ECCE", "MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    "bioinformatics":("ECCE", "MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    "mtr":              ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    "mechatronics":     ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    "robotics":         ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    "ase":              ("IDE",  "ASE", "✈️  Aerospace Engineering (ASE)"),
    "aerospace":        ("IDE",  "ASE", "✈️  Aerospace Engineering (ASE)"),
    "mse":              ("IDE",  "MSE", "🔬 Materials Science and Engineering (MSE)"),
    "materials":        ("IDE",  "MSE", "🔬 Materials Science and Engineering (MSE)"),
    "ime":              ("IDE",  "IME", "🏭 Industrial & Manufacturing Engineering (IME)"),
    "industrial":       ("IDE",  "IME", "🏭 Industrial & Manufacturing Engineering (IME)"),
    "manufacturing":    ("IDE",  "IME", "🏭 Industrial & Manufacturing Engineering (IME)"),
    "cpe":              ("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
    "chemical":         ("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
    "petrochemical":    ("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
    "env":              ("EECE", "ENV", "🌿 Environmental Engineering (ENV)"),
    "environmental":    ("EECE", "ENV", "🌿 Environmental Engineering (ENV)"),
    "eme":              ("EECE", "EME", "🏗️  Electromechanical Buildings Systems (EME)"),
    "electromechanical":("EECE", "EME", "🏗️  Electromechanical Buildings Systems (EME)"),
    "mpe":              ("EECE", "MPE", "⚙️  Mechanical Power & Energy Engineering (MPE)"),
    "mechanical":       ("EECE", "MPE", "⚙️  Mechanical Power & Energy Engineering (MPE)"),
}

TRACK_MENU = (
    "Please choose your track by typing one of the keywords:\n\n"
    "| Track | Department | Say |\n"
    "| :--- | :--- | :--- |\n"
    "| 💻 Computer Engineering | CSE (ECCE) | `CSE` or `computer` |\n"
    "| 📡 Electronics & Communications | ECE (ECCE) | `ECE` or `communications` |\n"
    "| ⚡ Electrical Power | EPE (ECCE) | `EPE` or `power` |\n"
    "| 🧬 Biomedical & Bioinformatics | MIE (ECCE) | `MIE` or `biomedical` |\n"
    "| 🤖 Mechatronics | MTR (IDE) | `MTR` or `mechatronics` |\n"
    "| ✈️ Aerospace | ASE (IDE) | `ASE` or `aerospace` |\n"
    "| 🔬 Materials Science | MSE (IDE) | `MSE` or `materials` |\n"
    "| 🏭 Industrial & Manufacturing | IME (IDE) | `IME` or `industrial` |\n"
    "| ⚗️ Chemical & Petrochemical | CPE (EECE) | `CPE` or `chemical` |\n"
    "| 🌿 Environmental | ENV (EECE) | `ENV` or `environmental` |\n"
    "| 🏗️ Electromechanical Buildings | EME (EECE) | `EME` or `electromechanical` |\n"
    "| ⚙️ Mechanical Power & Energy | MPE (EECE) | `MPE` or `mechanical` |"
)

def detect_track(text: str) -> tuple | None:
    lower = text.lower()
    for kw, info in TRACK_MAP.items():
        if re.search(rf'\b{re.escape(kw)}\b', lower):
            return info
    return None

# ════════════════════════════════════════════════════════════════
# 4. COURSE CATALOG (unchanged)
# ════════════════════════════════════════════════════════════════

def build_catalog(data: dict) -> dict:
    cat = {}
    shared_opts = data['curriculum'].get('shared_elective_options', {})

    def resolve_options(c):
        """Options can be a string ref like 'LRA_elective_1' or an inline list."""
        opts = c.get('options', [])
        if isinstance(opts, str):
            return shared_opts.get(opts, [])
        if isinstance(opts, list):
            return opts
        return []

    def get_ch(c):
        """Handle both 'credit hours' (space) and 'credit_hours' (underscore)."""
        val = c.get('credit hours') or c.get('credit_hours') or 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return 0

    def add(c, source):
        code = (c.get('code') or '').strip()
        if not code or code in cat:
            return
        cat[code] = {
            'code':    code,
            'name':    c.get('name', ''),
            'ch':      get_ch(c),
            'prereq':  c.get('prereq') or None,
            'type':    (c.get('Type') or c.get('type') or '').lower().strip(),
            'source':  source,
            'options': resolve_options(c),
        }

    found = data['curriculum']['PHASE_1_FOUNDATION']
    for sem in ['semester_1', 'semester_2', 'semester_3']:
        for c in found.get(sem, []):
            add(c, f'Foundation/{sem}')

    for sk, school in data['curriculum']['PHASE_2_SCHOOLS'].items():
        for c in school.get('semester_4_core', []):
            add(c, f'{sk}/semester_4')
        for dk, dept in school.get('departments', {}).items():
            for semk, courses in dept.get('semesters', {}).items():
                for c in courses:
                    add(c, f'{sk}/{dk}/{semk}')
    return cat

CATALOG = build_catalog(ejust_data) if ejust_data else {}

def trace_chain(code: str, cat: dict, visited: set = None) -> list:
    if visited is None:
        visited = set()
    if not code or code not in cat or code in visited:
        return []
    visited.add(code)
    c     = cat[code]
    chain = trace_chain(c.get('prereq'), cat, visited)
    chain.append(c)
    return chain

def get_track_prereqs(school: str, dept: str) -> dict:
    if not ejust_data:
        return {}

    school_data   = ejust_data['curriculum']['PHASE_2_SCHOOLS'].get(school, {})
    dept_data     = school_data.get('departments', {}).get(dept, {})
    sems          = dept_data.get('semesters', {})
    entry_courses = [c for c in sems.get('semester_5', []) if c.get('code')]
    sem4_all      = school_data.get('semester_4_core', [])
    sem4_courses  = [c for c in sem4_all if c.get('Type', '').lower() in ('core', 'school')]

    seen_codes  = set()
    all_prereqs = []

    src_courses = entry_courses + [c for c in sem4_all if c.get('Type', '').lower() == 'school']
    for ec in src_courses:
        for step in trace_chain(ec['code'], CATALOG):
            if step['code'] not in seen_codes and step['code'] != ec['code']:
                seen_codes.add(step['code'])
                all_prereqs.append(step)

    found_data = ejust_data['curriculum']['PHASE_1_FOUNDATION']
    sem3_core_elective_prereqs: dict[str, dict] = {}

    for c3 in found_data.get('semester_3', []):
        if c3.get('Type', '').lower() != 'core':
            continue
        prereq_code = c3.get('prereq')
        if not prereq_code:
            continue
        prereq_course = CATALOG.get(prereq_code)
        if not prereq_course:
            continue
        if prereq_course.get('type') == 'elective' and 'Foundation' in prereq_course.get('source', ''):
            sem3_core_elective_prereqs[c3['code']] = {
                'course':     c3,
                'blocked_by': prereq_course,
            }
            if prereq_code not in seen_codes:
                seen_codes.add(prereq_code)
                all_prereqs.append(prereq_course)

    elective_prereqs = [c for c in all_prereqs
                        if 'Foundation' in c.get('source', '') and c['type'] == 'elective']

    sem3_impact: dict[str, dict] = {}
    sem3_impact.update(sem3_core_elective_prereqs)

    for c3 in found_data.get('semester_3', []):
        if c3.get('Type', '').lower() == 'core' and c3.get('prereq'):
            blocked_by = [ep for ep in elective_prereqs if ep['code'] == c3['prereq']]
            if blocked_by and c3['code'] not in sem3_impact:
                sem3_impact[c3['code']] = {
                    'course':     c3,
                    'blocked_by': blocked_by[0],
                }

    return {
        'entry_courses':    entry_courses,
        'sem4_courses':     sem4_courses,
        'all_prereqs':      all_prereqs,
        'elective_prereqs': elective_prereqs,
        'sem3_impact':      sem3_impact,
    }

def load_semester(school: str, dept: str, sem_num: str) -> tuple[list, str]:
    if not ejust_data:
        return [], "Data unavailable"

    found   = ejust_data['curriculum']['PHASE_1_FOUNDATION']
    schools = ejust_data['curriculum']['PHASE_2_SCHOOLS']
    shared_opts = ejust_data['curriculum'].get('shared_elective_options', {})

    def normalize(c):
        """Normalize a course dict — unify credit hours field and resolve options."""
        c = dict(c)
        # Unify credit hours
        if 'credit_hours' in c and 'credit hours' not in c:
            c['credit hours'] = c['credit_hours']
        # Unify Type field
        if 'type' in c and 'Type' not in c:
            c['Type'] = c['type']
        # Resolve string options
        opts = c.get('options', [])
        if isinstance(opts, str):
            c['options'] = shared_opts.get(opts, [])
        return c

    if sem_num in ('1', '2', '3'):
        sem_key = f'semester_{sem_num}'
        courses = [normalize(c) for c in found.get(sem_key, [])]
        title   = f'Foundation — Semester {sem_num}'
        return courses, title

    if sem_num == '4':
        school_data = schools.get(school, {})
        courses = [normalize(c) for c in school_data.get('semester_4_core', [])]
        title   = f'{school} — Semester 4 (School Core)'
        return courses, title

    sem_key     = f'semester_{sem_num}'
    school_data = schools.get(school, {})
    dept_data   = school_data.get('departments', {}).get(dept, {})
    courses     = [normalize(c) for c in dept_data.get('semesters', {}).get(sem_key, []) if c.get('code')]
    title       = f'{dept} — Semester {sem_num}'
    return courses, title

def workload_check(total_ch: int, limit: int) -> str:
    if total_ch > limit:
        return (
            f"⚠️ **OVER LIMIT:** {total_ch} CH registered but limit is {limit} CH. "
            f"Must drop {total_ch - limit} CH."
        )
    return f"✅ Within limit: {total_ch} / {limit} CH."

def ctx_ask_track(max_ch: int) -> str:
    return (
        "[ADVISOR INSTRUCTION — NO TRACK SELECTED]\n"
        f"Student's credit limit: {max_ch} CH.\n\n"
        "SENPAI MUST ask the student which track/department they want to join "
        "BEFORE giving any course or semester advice. "
        "Present the track menu below and wait for their answer.\n\n"
        f"{TRACK_MENU}\n\n"
        "Do NOT show any courses, schedule, or semester info until they choose."
    )


def ctx_track_overview(school: str, dept: str, label: str,
                        max_ch: int, is_half: bool) -> str:
    info = get_track_prereqs(school, dept)
    if not info:
        return f"[Could not load prereq data for {label}]"

    if info['elective_prereqs']:
        lines = []
        for ep in info['elective_prereqs']:
            sem_label = ep['source'].replace('Foundation/', 'Semester ').replace('semester_', '')
            lines.append(
                f"  🚨 {ep['code']} — {ep['name']} ({ep['ch']} CH) "
                f"[listed as ELECTIVE in {sem_label}]"
            )
        elec_block = (
            "━━━ ⚠️  ELECTIVE COURSES THAT ARE ACTUALLY PREREQUISITES ━━━\n"
            "These appear as optional electives in foundation semesters, but they ARE\n"
            "prerequisites for core courses. Skipping them will block your track entry!\n"
            + "\n".join(lines)
        )
    else:
        elec_block = "━━━ ELECTIVE PREREQUISITES ━━━\n  None — no electives are hard prerequisites for this track."

    sem3_block = ""
    if info['sem3_impact']:
        lines = []
        for code, imp in info['sem3_impact'].items():
            c3  = imp['course']
            blk = imp['blocked_by']
            lines.append(
                f"  ❌ If you skip {blk['code']} ({blk['name']}) → "
                f"you CANNOT take {c3.get('code')} ({c3.get('name')}) in Semester 3, "
                f"which is a CORE course!"
            )
        sem3_block = (
            "━━━ 💥 SEM3 CORE COURSES THAT WILL BE BLOCKED ━━━\n"
            + "\n".join(lines)
        )

    sem4_lines = "\n".join([
        f"  [{c.get('Type','?')}] {c['code']} — {c['name']} "
        f"({c.get('credit hours','?')} CH) | prereq: {c.get('prereq','None')}"
        for c in info['sem4_courses']
    ]) or "  (none listed)"

    sem5_lines = "\n".join([
        f"  ➡️  {c['code']} — {c['name']} | prereq: {c.get('prereq','None')}"
        for c in info['entry_courses']
    ]) or "  (no data yet)"

    half_warn = ""
    if is_half:
        elec_codes = [ep['code'] for ep in info['elective_prereqs']]
        half_warn = (
            "\n━━━ 🔴 HALF-LOAD STUDENT ALERT ━━━\n"
            f"You are on Academic Probation (max {max_ch} CH).\n"
            f"Despite the tight budget, you MUST fit these elective prerequisites into your schedule:\n"
            + "\n".join(f"  • {c}" for c in elec_codes)
            + "\nSkipping them delays your track entry by a FULL YEAR."
            if elec_codes else (
                "\n━━━ HALF-LOAD STUDENT NOTE ━━━\n"
                f"You are on Academic Probation (max {max_ch} CH). "
                "Good news: no hidden elective prerequisites for this track."
            )
        )

    return f"""
[TRACK SELECTED: {label}]
School: {school} | Dept: {dept} | Credit Limit: {max_ch} CH

{elec_block}

{sem3_block}

━━━ SEMESTER 4 GATEWAY COURSES (must pass all to enter track) ━━━
{sem4_lines}

━━━ SEMESTER 5 ENTRY COURSES ━━━
{sem5_lines}
{half_warn}

INSTRUCTION FOR SENPAI:
1. Confirm the student's track choice warmly.
2. Immediately highlight any elective prerequisites — these are the most common mistake.
3. Explain the sem3 impact: skipping that elective blocks a core course.
4. Show the semester-4 gateway and semester-5 entry so they know the full path.
5. If half-load, stress they MUST budget for the elective prereqs within 14 CH.
6. Encourage them: knowing the path early gives a huge advantage.
""".strip()


def ctx_semester_plan(courses: list, title: str, sem_num: str,
                       school: str, dept: str, label: str,
                       max_ch: int, is_half: bool) -> str:
    track_info        = get_track_prereqs(school, dept)
    elec_prereq_codes = {ep['code'] for ep in track_info.get('elective_prereqs', [])}

    core_courses = [c for c in courses if (c.get('Type','').lower() in ('core','school')) and c.get('code')]
    elec_courses = [c for c in courses if c.get('Type','').lower() == 'elective' and c.get('code')]
    core_ch      = sum(int(c.get('credit hours') or 0) for c in core_courses)
    total_ch     = sum(int(c.get('credit hours') or 0) for c in courses if c.get('code'))

    def annotate(c):
        ch      = int(c.get('credit hours') or 0)
        code    = c.get('code','?')
        name    = c.get('name','?')
        prereq  = c.get('prereq') or 'None'
        ctype   = c.get('Type','?')
        tag     = " 🔑 [TRACK PREREQ — must not skip]" if code in elec_prereq_codes else ""
        opts    = c.get('options', [])
        opt_str = (" (choose one: " + ", ".join(o['name'] for o in opts) + ")") if opts else ""
        return f"  [{ctype}] {code} — {name}{opt_str} ({ch} CH) | prereq: {prereq}{tag}"

    all_lines = "\n".join(annotate(c) for c in courses if c.get('code'))
    safety    = workload_check(total_ch, max_ch)

    if not is_half:
        return (
            f"[SEMESTER PLAN — {title} | Track: {label}]\n"
            f"Status: Regular/Over-Achiever | Limit: {max_ch} CH\n"
            f"{safety}\n\n"
            f"Courses:\n{all_lines}\n\n"
            f"Total: {total_ch} CH"
        )

    budget     = max_ch - core_ch
    elec_sorted = sorted(elec_courses,
                         key=lambda c: (0 if c.get('code') in elec_prereq_codes else 1,
                                        int(c.get('credit hours') or 0)))

    recommended, deferred, running = [], [], 0
    for e in elec_sorted:
        ch = int(e.get('credit hours') or 0)
        if running + ch <= budget:
            recommended.append(e); running += ch
        else:
            deferred.append(e)

    deferred_track_prereqs = [e for e in deferred if e.get('code') in elec_prereq_codes]
    core_lines = "\n".join(annotate(c) for c in core_courses) or "  (none)"
    rec_lines  = "\n".join(annotate(e) for e in recommended)  or "  (no elective budget remaining)"
    def_lines  = "\n".join(annotate(e) for e in deferred)     or "  None"

    if budget == 0:
        budget_note = f"📌 Core courses exactly fill the 14 CH limit. No elective budget this semester."
    elif budget > 0:
        budget_note = f"📌 After core ({core_ch} CH), you have {budget} CH left for electives. Track prerequisites were picked first."
    else:
        budget_note = f"⚠️ CRITICAL: Core courses ({core_ch} CH) already EXCEED the 14 CH limit! Contact academic office."

    prereq_warn = ""
    if deferred_track_prereqs:
        names = ", ".join(f"{e['code']} ({e['name']})" for e in deferred_track_prereqs)
        prereq_warn = (
            f"\n🚨 URGENT: Track-prerequisite elective(s) couldn't fit this semester:\n"
            f"  {names}\n"
            f"You MUST take these next semester — they block your sem3/4 core courses!"
        )

    return f"""
[HALF-LOAD SEMESTER PLAN — {title} | Track: {label}]
Status: Academic Probation | Limit: {max_ch} CH | Plan Total: {core_ch + running} CH
{budget_note}
{safety}

━━━ 🔵 MANDATORY CORE (cannot be dropped) ━━━
{core_lines}

━━━ ✅ RECOMMENDED ELECTIVES (within budget, track prereqs first) ━━━
{rec_lines}

━━━ ⏳ DEFERRED ELECTIVES (take when CGPA recovers) ━━━
{def_lines}
{prereq_warn}

INSTRUCTION FOR SENPAI:
1. Show the must-take core courses clearly — non-negotiable.
2. For each recommended elective, state if it is a track prerequisite and exactly why it matters.
3. For any deferred track prereqs, give a clear warning they must take it next semester.
4. Motivate: pass everything → CGPA 2.0+ → Regular Load (19 CH) unlocked next semester.
5. Tone: supportive coach, not alarming.
""".strip()


# ════════════════════════════════════════════════════════════════
# 9. PAGE CONFIG & SESSION STATE
# ════════════════════════════════════════════════════════════════

##st.set_page_config(page_title="Senpai — E-JUST Advisor", layout="wide", page_icon="🎓")
##st.title("🎓 Senpai — E-JUST Academic Advisor")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-hc-v1-3d0b21306ebd475c92404d9870d890a39a0c4a6a345945f7a5287bd75c595050")
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url="https://ai.hackclub.com/proxy/v1"
)


if "messages"   not in st.session_state: st.session_state.messages   = []
if "user_cgpa"  not in st.session_state: st.session_state.user_cgpa  = None
if "track_info" not in st.session_state: st.session_state.track_info = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ════════════════════════════════════════════════════════════════
# 10. MAIN CHAT HANDLER
# ════════════════════════════════════════════════════════════════

if prompt := st.chat_input("Ask Senpai …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        # ── A. EXTRACT & PERSIST CGPA ────────────────────────────────────
        gpa_match = re.search(r'\b(?:cgpa|gpa)\s*[:=]?\s*(\d(?:\.\d+)?)\b', prompt.lower())
        if not gpa_match:
            gpa_match = re.search(r'\b([0-3](?:\.\d+)?)\b', prompt)
        if gpa_match:
            candidate = float(gpa_match.group(1))
            if 0.0 <= candidate <= 4.0:
                st.session_state.user_cgpa = candidate

        user_cgpa  = st.session_state.user_cgpa if st.session_state.user_cgpa is not None else 3.0
        status_lbl, max_ch = get_student_status(user_cgpa)

        # ── UPDATE: derive is_half from status label (so if we remapped <2.0 to Regular, is_half becomes False)
        is_half = True if "Half" in status_lbl or "Half-Load" in status_lbl else False

        # ── B. DETECT & PERSIST TRACK ─────────────────────────────────────
        detected = detect_track(prompt)
        if detected:
            st.session_state.track_info = detected
        track_info  = st.session_state.track_info
        school      = track_info[0] if track_info else None
        dept        = track_info[1] if track_info else None
        track_label = track_info[2] if track_info else "Not chosen yet"

        # ── C. PDF RAG — only for handbook/rules questions, never for courses ──
        pdf_ctx = ""
        _pl = prompt.lower()
        if vdb and not any(kw in _pl for kw in [
            'course', 'semester', 'courses', 'curriculum', 'subject',
            'credit', 'prereq', 'prerequisite', 'schedule', 'plan'
        ]):
            docs    = vdb.similarity_search(prompt, k=3)
            pdf_ctx = "\n".join(d.page_content for d in docs)

        # ── D. BUILD ADVISOR CONTEXT ──────────────────────────────────────
        p_lower   = prompt.lower()
        sem_match = re.search(r'\b(?:semester|sem)\s*(\d)\b', p_lower)
        adv_ctx   = ""

        sem_num            = sem_match.group(1) if sem_match else None
        foundation_sem     = sem_num in ('1', '2', '3')
        track_specific_sem = sem_num in ('4', '5', '6', '7', '8')

        needs_track = (
            track_specific_sem
            or (not sem_num and any(kw in p_lower for kw in [
                'schedule', 'plan', 'recommend', 'which courses', 'what courses',
                'my track', 'which track', 'what track', 'track advice',
                'graduation', 'department', 'which department',
            ]))
        )

        if needs_track and not track_info:
            adv_ctx = ctx_ask_track(max_ch)
            if sem_num:
                adv_ctx += (
                    f"\n\nNOTE: Student asked about Semester {sem_num}. "
                    f"This semester is track-specific. Once they choose a track, "
                    f"immediately show their semester {sem_num} plan."
                )

        elif sem_num:
            if foundation_sem:
                courses, title = load_semester(school or 'ECCE', dept or 'CSE', sem_num)
                if courses:
                    adv_ctx = ctx_semester_plan(
                        courses, title, sem_num,
                        school or 'ECCE', dept or 'CSE', track_label,
                        max_ch, is_half
                    )
                    if not track_info:
                        adv_ctx += (
                            "\n\n[SOFT NUDGE FOR SENPAI]: After answering, casually ask "
                            "which track the student plans to pursue."
                        )
                else:
                    adv_ctx = f"[No data found for Foundation Semester {sem_num}]"

            elif track_info:
                courses, title = load_semester(school, dept, sem_num)
                if courses:
                    adv_ctx = ctx_semester_plan(
                        courses, title, sem_num,
                        school, dept, track_label,
                        max_ch, is_half
                    )
                else:
                    adv_ctx = (
                        f"[No course data yet for {track_label} Semester {sem_num}. "
                        f"This department's curriculum may still be incomplete.]"
                    )
            else:
                adv_ctx = ctx_ask_track(max_ch)
                adv_ctx += (
                    f"\n\nNOTE: Student asked about Semester {sem_num} which varies by track. "
                    f"Ask for their track first, then show the semester plan."
                )

        elif track_info:
            adv_ctx = ctx_track_overview(school, dept, track_label, max_ch, is_half)

        else:
            adv_ctx = (
                "[GENERAL QUESTION — No semester or track-specific data required]\n"
                "Answer the student's question using the handbook context and your general "
                "E-JUST knowledge. You may mention the available tracks and invite them to "
                "share their track if relevant, but do NOT block the answer."
            )

        # ── E. PROFESSOR CONTEXT ──────────────────────────────────────────
        prof_ctx  = ""
        asks_prof = any(kw in p_lower for kw in [
            'professor', 'prof', 'doctor', 'dr ', 'dr.', 'instructor',
            'email', 'office', 'contact', 'research', 'faculty',
            'who teaches', 'who is', 'find professor', 'department professor',
            'مين', 'دكتور', 'استاذ'
        ])

        if profs_df is not None:
            # ── Try to match a specific professor name ────────────────────
            matched_rows = []
            query_words = [w for w in p_lower.split() if len(w) > 2]

            scored = []
            for _, row in profs_df.iterrows():
                pname = str(row.get('Name', '')).lower()
                pname_parts = [p for p in pname.split() if len(p) > 2]
                score = 0
                # Score: how many query words appear in professor name
                score += sum(1 for qw in query_words if qw in pname) * 2
                # Score: how many professor name parts appear in query
                score += sum(1 for pp in pname_parts if pp in p_lower)
                if score > 0:
                    scored.append((score, row))

            if scored:
                # Sort by score descending
                scored.sort(key=lambda x: x[0], reverse=True)
                top_score = scored[0][0]
                # Only keep rows within 1 point of the best score
                matched_rows = [row for s, row in scored if s >= top_score - 1]
                # Cap at 5 results to avoid flooding
                matched_rows = matched_rows[:5]

            # ── Try to match by department ────────────────────────────────
            if not matched_rows:
                dept_keywords = {
                    'computer': 'Computer', 'cse': 'Computer',
                    'mechatronics': 'Mechatronics', 'mtr': 'Mechatronics', 'robotics': 'Mechatronics',
                    'aerospace': 'Aerospace', 'ase': 'Aerospace',
                    'materials': 'Materials', 'mse': 'Materials',
                    'industrial': 'Industrial', 'manufacturing': 'Industrial', 'ime': 'Industrial',
                    'energy': 'Energy', 'ere': 'Energy', 'mpe': 'Energy',
                    'chemical': 'Chemical', 'cpe': 'Chemical',
                    'electrical': 'Electrical', 'epe': 'Electrical',
                    'environmental': 'Environmental', 'env': 'Environmental',
                    'biomedical': 'Biomedical', 'mie': 'Biomedical',
                    'electronics': 'Electronics', 'ece': 'Electronics',
                    'accounting': 'Accounting', 'business': 'Business',
                    'architecture': 'Architecture', 'art': 'Art & Design',
                }
                for kw, dept_kw in dept_keywords.items():
                    if kw in p_lower:
                        for _, row in profs_df.iterrows():
                            dept_val = str(row.get('Department', '')).lower()
                            faculty_val = str(row.get('Faculty', '')).lower()
                            if dept_kw.lower() in dept_val or dept_kw.lower() in faculty_val:
                                matched_rows.append(row)
                        break

            if matched_rows:
                # Format matched professors
                parts = []
                for row in matched_rows:
                    name     = row.get('Name', 'Unknown')
                    title    = row.get('Job Title', 'N/A')
                    dept     = row.get('Department', 'N/A')
                    faculty  = row.get('Faculty', 'N/A')
                    office   = row.get('Office Location', 'N/A')
                    email    = row.get('Email', 'N/A')
                    research = row.get('Research Fields', 'N/A')
                    parts.append(
                        f"• {name} | {title}\n"
                        f"  Department: {dept} | Faculty: {faculty}\n"
                        f"  Office: {office} | Email: {email}\n"
                        f"  Research: {research}"
                    )
                prof_ctx = "[PROFESSOR DATA]\n" + "\n\n".join(parts)

            elif asks_prof:
                # General professor query — provide full list
                rows = []
                for _, row in profs_df.iterrows():
                    name   = row.get('Name', 'Unknown')
                    title  = row.get('Job Title', 'N/A')
                    dept   = row.get('Department', 'N/A')
                    email  = row.get('Email', 'N/A')
                    rows.append(f"  • {name} | {title} | Dept: {dept} | {email}")
                prof_ctx = "[ALL PROFESSORS]\n" + "\n".join(rows)

        # ── F. MISSION DETECTION ──────────────────────────────────────────
        asks_registration = any(kw in p_lower for kw in [
            'register', 'registration', 'add course', 'drop course', 'enroll',
            'sign up', 'how to register', 'course registration', 'add/drop',
            'portal', 'student system', 'sis', 'how do i add'
        ])
        asks_handbook = any(kw in p_lower for kw in [
            'rule', 'policy', 'regulation', 'graduate', 'graduation', 'gpa requirement',
            'academic', 'probation', 'dismissal', 'appeal', 'leave', 'transfer',
            'credit', 'handbook', 'bylaw', 'attendance', 'exam', 'retake', 'withdraw'
        ])
        asks_schedule = any(kw in p_lower for kw in [
            'schedule', 'plan', 'roadmap', 'semester', 'courses', 'credit hours',
            'what should i take', 'which courses', 'next semester'
        ])

        active_mission = (
            "COURSE REGISTRATION ASSISTANCE"      if asks_registration else
            "PROFESSOR REVIEWS & RECOMMENDATIONS"  if (asks_prof and prof_ctx) else
            "HANDBOOK / ACADEMIC RULES"            if asks_handbook else
            "COURSE SCHEDULE & PLANNING"           if asks_schedule else
            "GENERAL ADVISING"
        )

        # ── G. SYSTEM PROMPT ──────────────────────────────────────────────
        system_prompt = f"""
You are **Senpai**, the official AI academic advisor for E-JUST (Egypt-Japan University of Science and Technology).
You are friendly, direct, and trustworthy. Students depend on you for accurate information.

━━━ YOUR MISSIONS ━━━
1. 📋 COURSE REGISTRATION HELP — explain how to register/add/drop, warn about prereqs and limits
2. 👨‍🏫 PROFESSOR INFO — answer questions about professors using ONLY the PROFESSOR DATA section below.
   Available info per professor: Name, Job Title, Department, Faculty, Office Location, Email, Research Fields.
   If asked about a professor not in the data, say they are not in the database.
3. 📖 HANDBOOK & ACADEMIC RULES — answer policy/GPA/graduation questions from handbook only
4. 🗓️ COURSE PLANNING — help plan semesters using ONLY the course data below

━━━ ACTIVE MISSION ━━━
{active_mission}

━━━ STUDENT PROFILE ━━━
  • CGPA: {user_cgpa} | Status: {status_lbl} | Limit: {max_ch} CH | Track: {track_label}



━━ CREDIT RULES ━━━
  • CGPA < 2.0 → Half-Load — max 14 CH
  • 2.0–2.99   → Regular   — max 19 CH
  • ≥ 3.0      → Honors    — max 21 CH

━━━ 🚨 ABSOLUTE RULES — NEVER BREAK THESE ━━━
1. COURSE DATA: Every single course name, code, credit hours, and prerequisite you mention
   MUST exist VERBATIM in the COURSE & SCHEDULE DATA section below.
   If it is not there → it does not exist → do NOT mention it.

2. NO HALLUCINATION: NEVER invent, estimate, guess, or infer courses.
   NEVER say "typical courses include..." or "based on standard curricula..."
   NEVER create tables of made-up courses.
   NEVER use the handbook to answer course questions.

3. IF DATA IS MISSING: If a semester has no data in COURSE & SCHEDULE DATA,
   say EXACTLY: "I don't have course data for [track] Semester [N] yet.
   Only the data in Tracks.json is available — I cannot invent what's missing."
   Then STOP. Do not add estimates, suggestions, or handbook guesses.

4. HANDBOOK: Use ONLY for academic rules, policies, GPA regulations, graduation requirements.
   NEVER for course lists, course names, credit hours, or prerequisites.

5. PROFESSOR DATA: Only mention professors that appear in PROFESSOR DATA below.

━━━ COURSE & SCHEDULE DATA (ONLY SOURCE FOR COURSES) ━━━
{adv_ctx}

━━━ HANDBOOK CONTEXT (ONLY FOR RULES/POLICIES) ━━━
{pdf_ctx}

━━━ PROFESSOR DATA ━━━
{prof_ctx if prof_ctx else "No professor data matched this query."}
""".strip()

        # ── H. CALL OPENAI ────────────────────────────────────────────────
        try:
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            resp = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    *history,
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            answer = resp.choices[0].message.content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"OpenRouter API Error: {e}")
