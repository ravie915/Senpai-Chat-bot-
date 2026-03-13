import streamlit as st
import json
import pandas as pd
import os
import re
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

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
# ════════════════════════════════════════════════════════════════

def get_student_status(cgpa: float) -> tuple[str, int]:
    """
    CGPA < 2.0  → Half-Load   (Academic Probation)  → max 14 CH
    2.0–2.99    → Regular Load                       → max 19 CH
    ≥ 3.0       → Over-Achiever                      → max 21 CH
    """
    if cgpa < 2.0:
        return "Half-Load (Academic Probation)", 14
    elif cgpa < 3.0:
        return "Regular Load", 19
    else:
        return "Over-Achiever (Honors)", 21

# ════════════════════════════════════════════════════════════════
# 3. TRACK MAP  — keyword → (school, dept, display label)
# ════════════════════════════════════════════════════════════════

TRACK_MAP: dict[str, tuple[str, str, str]] = {
    # CSE / Computer Engineering
    "cse":          ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    "computer":     ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    "software":     ("ECCE", "CSE", "💻 Computer Engineering (CSE)"),
    # ECE / Electronics & Communications
    "ece":          ("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "electronics":  ("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    "communications":("ECCE","ECE", "📡 Electronics & Communications (ECE)"),
    "communication":("ECCE", "ECE", "📡 Electronics & Communications (ECE)"),
    # EPE / Electrical Power
    "epe":          ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    "power":        ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    "electrical":   ("ECCE", "EPE", "⚡ Electrical Power Engineering (EPE)"),
    # MIE / Biomedical
    "mie":          ("ECCE", "MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    "biomedical":   ("ECCE", "MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    "bioinformatics":("ECCE","MIE", "🧬 Biomedical & Bioinformatics Engineering (MIE)"),
    # MTR / Mechatronics
    "mtr":          ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    "mechatronics": ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    "robotics":     ("IDE",  "MTR", "🤖 Mechatronics Engineering (MTR)"),
    # CPE / Chemical & Petrochemical
    "cpe":          ("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
    "chemical":     ("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
    "petrochemical":("EECE", "CPE", "⚗️  Chemical & Petrochemical Engineering (CPE)"),
}

TRACK_MENU = (
    "Please choose your track by typing one of the keywords:\n\n"
    "| Track | Department | Say |\n"
    "| :--- | :--- | :--- |\n"
    "| 💻 Computer Engineering | CSE (ECCE) | `CSE` or `computer` |\n"
    "| 📡 Electronics & Communications | ECE (ECCE) | `ECE` or `communications` |\n"
    "| ⚡ Electrical Power | EPE (ECCE) | `EPE` or `power` |\n"
    "| 🧬 Biomedical & Bioinformatics | MIE (ECCE) | `MIE` or `biomedical` |\n"
    "| 🤖 Mechatronics / Robotics | MTR (IDE) | `MTR` or `robotics` |\n"
    "| ⚗️ Chemical & Petrochemical | CPE (EECE) | `CPE` or `chemical` |"
)

def detect_track(text: str) -> tuple | None:
    """Scan message for any TRACK_MAP keyword. Returns (school, dept, label) or None."""
    lower = text.lower()
    for kw, info in TRACK_MAP.items():
        if re.search(rf'\b{re.escape(kw)}\b', lower):
            return info
    return None

# ════════════════════════════════════════════════════════════════
# 4. COURSE CATALOG  — flat lookup of every course in the JSON
# ════════════════════════════════════════════════════════════════

def build_catalog(data: dict) -> dict:
    cat = {}
    def add(c, source):
        code = (c.get('code') or '').strip()
        if not code or code in cat:
            return
        cat[code] = {
            'code':    code,
            'name':    c.get('name', ''),
            'ch':      int(c.get('credit hours') or 0),
            'prereq':  c.get('prereq') or None,
            'type':    (c.get('Type') or '').lower().strip(),
            'source':  source,
            'options': c.get('options', []),
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

# ════════════════════════════════════════════════════════════════
# 5. PREREQUISITE ENGINE
# ════════════════════════════════════════════════════════════════

def trace_chain(code: str, cat: dict, visited: set = None) -> list:
    """Walk prereqs backward and return ordered chain: root → code."""
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
    """
    For a track identified by (school, dept):
      - Find all semester-5 entry courses + semester-4 school courses
      - Trace full prereq chains back to foundation semesters
      - ALSO scan all semester-3 core courses for elective prerequisites
        (e.g. CSE 211 is a sem3 Core but requires CSE 111 which is a sem2 Elective)
      - Identify elective_prereqs: foundation ELECTIVE courses that are hard prereqs
      - Identify sem3_impact: sem3 core courses blocked if the elective is skipped

    Returns:
        entry_courses    — sem-5 courses for this dept
        sem4_courses     — school-wide sem-4 core/school courses
        all_prereqs      — every prereq course found in chains
        elective_prereqs — foundation ELECTIVE courses that are prerequisites
        sem3_impact      — {blocked_code: {course, blocked_by}} for sem3 cores at risk
    """
    if not ejust_data:
        return {}

    school_data   = ejust_data['curriculum']['PHASE_2_SCHOOLS'].get(school, {})
    dept_data     = school_data.get('departments', {}).get(dept, {})
    sems          = dept_data.get('semesters', {})
    entry_courses = [c for c in sems.get('semester_5', []) if c.get('code')]

    sem4_all     = school_data.get('semester_4_core', [])
    sem4_courses = [c for c in sem4_all if c.get('Type', '').lower() in ('core', 'school')]

    # ── Step 1: Trace chains from sem5 entry + sem4 school courses ──────────
    seen_codes  = set()
    all_prereqs = []

    src_courses = entry_courses + [c for c in sem4_all if c.get('Type','').lower() == 'school']
    for ec in src_courses:
        for step in trace_chain(ec['code'], CATALOG):
            if step['code'] not in seen_codes and step['code'] != ec['code']:
                seen_codes.add(step['code'])
                all_prereqs.append(step)

    # ── Step 2: Scan ALL sem3 core courses for elective prerequisites ────────
    # This catches cases like CSE 211 (sem3 Core) ← CSE 111 (sem2 Elective)
    # even if CSE 211 itself is not in the sem5 chain of this specific track.
    # Reason: sem3 core courses are mandatory for ALL students, so an elective
    # that gates a sem3 core is effectively mandatory for everyone.
    found_data = ejust_data['curriculum']['PHASE_1_FOUNDATION']
    sem3_core_elective_prereqs: dict[str, dict] = {}  # {blocked_code: info}

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
            # Add this elective to all_prereqs if not already there
            if prereq_code not in seen_codes:
                seen_codes.add(prereq_code)
                all_prereqs.append(prereq_course)

    # ── Step 3: Collect elective_prereqs (what student must not skip) ────────
    elective_prereqs = [c for c in all_prereqs
                        if 'Foundation' in c.get('source', '') and c['type'] == 'elective']

    # ── Step 4: Build full sem3 impact map (union of both detection methods) ─
    sem3_impact: dict[str, dict] = {}

    # From sem3 core elective prereqs (step 2) — applies universally
    sem3_impact.update(sem3_core_elective_prereqs)

    # From chain tracing (step 1) — track-specific sem3 blocks
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

# ════════════════════════════════════════════════════════════════
# 6. SEMESTER DATA LOADER
# ════════════════════════════════════════════════════════════════

def load_semester(school: str, dept: str, sem_num: str) -> tuple[list, str]:
    """
    Returns (course_list, title) for any semester number.
    Handles: foundation (1-3), school sem4 core, dept semesters (5-8).
    """
    if not ejust_data:
        return [], "Data unavailable"

    found   = ejust_data['curriculum']['PHASE_1_FOUNDATION']
    schools = ejust_data['curriculum']['PHASE_2_SCHOOLS']

    if sem_num in ('1', '2', '3'):
        sem_key  = f'semester_{sem_num}'
        courses  = found.get(sem_key, [])
        title    = f'Foundation — Semester {sem_num}'
        return courses, title

    if sem_num == '4':
        school_data = schools.get(school, {})
        courses = school_data.get('semester_4_core', [])
        title   = f'{school} — Semester 4 (School Core)'
        return courses, title

    sem_key     = f'semester_{sem_num}'
    school_data = schools.get(school, {})
    dept_data   = school_data.get('departments', {}).get(dept, {})
    courses     = dept_data.get('semesters', {}).get(sem_key, [])
    title       = f'{dept} — Semester {sem_num}'
    return [c for c in courses if c.get('code')], title

# ════════════════════════════════════════════════════════════════
# 7. WORKLOAD SAFETY
# ════════════════════════════════════════════════════════════════

def workload_check(total_ch: int, limit: int) -> str:
    if total_ch > limit:
        return (
            f"⚠️ **OVER LIMIT:** {total_ch} CH registered but limit is {limit} CH. "
            f"Must drop {total_ch - limit} CH."
        )
    return f"✅ Within limit: {total_ch} / {limit} CH."

# ════════════════════════════════════════════════════════════════
# 8. CONTEXT BUILDERS  — these produce the text fed to the LLM
# ════════════════════════════════════════════════════════════════

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
    """
    Full prerequisite roadmap shown the moment a track is chosen.
    Covers: prereq chains, critical elective alerts, sem4 gateway, sem5 entry.
    """
    info = get_track_prereqs(school, dept)
    if not info:
        return f"[Could not load prereq data for {label}]"

    # ── Elective prereq alert block ─────────────────────────
    elec_block = ""
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

    # ── Sem3 impact block ───────────────────────────────────
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

    # ── Sem4 gateway ────────────────────────────────────────
    sem4_lines = "\n".join([
        f"  [{c.get('Type','?')}] {c['code']} — {c['name']} "
        f"({c.get('credit hours','?')} CH) | prereq: {c.get('prereq','None')}"
        for c in info['sem4_courses']
    ]) or "  (none listed)"

    # ── Sem5 entry ──────────────────────────────────────────
    sem5_lines = "\n".join([
        f"  ➡️  {c['code']} — {c['name']} | prereq: {c.get('prereq','None')}"
        for c in info['entry_courses']
    ]) or "  (no data yet)"

    # ── Half-load extra warning ─────────────────────────────
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
    """
    Semester-specific context. For half-load students, builds a tight
    14-CH plan that locks core first and prioritises elective prereqs.
    """
    track_info = get_track_prereqs(school, dept)
    elec_prereq_codes = {ep['code'] for ep in track_info.get('elective_prereqs', [])}

    # Separate core vs elective
    core_courses = [c for c in courses if (c.get('Type','').lower() in ('core','school')) and c.get('code')]
    elec_courses = [c for c in courses if c.get('Type','').lower() == 'elective' and c.get('code')]
    core_ch      = sum(int(c.get('credit hours') or 0) for c in core_courses)
    total_ch     = sum(int(c.get('credit hours') or 0) for c in courses if c.get('code'))

    # Annotate every course
    def annotate(c):
        ch     = int(c.get('credit hours') or 0)
        code   = c.get('code','?')
        name   = c.get('name','?')
        prereq = c.get('prereq') or 'None'
        ctype  = c.get('Type','?')
        tag    = " 🔑 [TRACK PREREQ — must not skip]" if code in elec_prereq_codes else ""
        opts   = c.get('options', [])
        opt_str = ""
        if opts:
            opt_str = " (choose one: " + ", ".join(o['name'] for o in opts) + ")"
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

    # ── HALF-LOAD: build tight 14-CH schedule ───────────────
    budget = max_ch - core_ch

    # Sort electives: track prereqs first, then smallest CH
    elec_sorted = sorted(elec_courses,
                         key=lambda c: (0 if c.get('code') in elec_prereq_codes else 1,
                                        int(c.get('credit hours') or 0)))

    recommended, deferred, running = [], [], 0
    for e in elec_sorted:
        ch = int(e.get('credit hours') or 0)
        if running + ch <= budget:
            recommended.append(e)
            running += ch
        else:
            deferred.append(e)

    # Check if any track-prereq elective was deferred (this is a warning)
    deferred_track_prereqs = [e for e in deferred if e.get('code') in elec_prereq_codes]

    core_lines = "\n".join(annotate(c) for c in core_courses) or "  (none)"
    rec_lines  = "\n".join(annotate(e) for e in recommended)  or "  (no elective budget remaining)"
    def_lines  = "\n".join(annotate(e) for e in deferred)     or "  None"

    # Budget note
    if budget == 0:
        budget_note = (
            f"📌 Core courses exactly fill the 14 CH limit. "
            f"No elective budget this semester — this is normal."
        )
    elif budget > 0:
        budget_note = (
            f"📌 After core ({core_ch} CH), you have {budget} CH left for electives. "
            f"Track prerequisites were picked first."
        )
    else:
        budget_note = (
            f"⚠️ CRITICAL: Core courses ({core_ch} CH) already EXCEED the 14 CH limit! "
            f"Student must contact academic office immediately."
        )

    # Deferred prereq warning
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
# 9. PAGE CONFIG & STREAMLIT SESSION STATE
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Senpai — E-JUST Advisor", layout="wide", page_icon="🎓")
st.title("🎓 Senpai — E-JUST Academic Advisor")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyDt-SMhkGuZne6DEW58zXLPo8J82kUTt30")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Persistent state
if "messages"     not in st.session_state: st.session_state.messages     = []
if "user_cgpa"    not in st.session_state: st.session_state.user_cgpa    = None
if "track_info"   not in st.session_state: st.session_state.track_info   = None
# track_info = None | (school, dept, label)

# Render chat history
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
        is_half    = user_cgpa < 2.0

        # ── B. DETECT & PERSIST TRACK ─────────────────────────────────────
        detected = detect_track(prompt)
        if detected:
            st.session_state.track_info = detected
        track_info  = st.session_state.track_info   # None | (school, dept, label)
        school      = track_info[0] if track_info else None
        dept        = track_info[1] if track_info else None
        track_label = track_info[2] if track_info else "Not chosen yet"

        # ── C. PDF RAG ────────────────────────────────────────────────────
        pdf_ctx = ""
        if vdb:
            docs    = vdb.similarity_search(prompt, k=3)
            pdf_ctx = "\n".join(d.page_content for d in docs)

        # ── D. BUILD ADVISOR CONTEXT ──────────────────────────────────────
        p_lower   = prompt.lower()
        sem_match = re.search(r'\b(?:semester|sem)\s*(\d)\b', p_lower)
        adv_ctx   = ""

        # ── Classify what the student is asking ───────────────────────────
        # Questions that REQUIRE knowing the track before answering:
        #   • Semester 4, 5, 6, 7, 8 (dept-specific content)
        #   • "my schedule", "my plan", "recommend courses", "which track", "what track"
        #   • Half-load schedule advice (needs track to prioritise electives)
        # Questions that do NOT require a track:
        #   • Semester 1, 2, 3 (shared foundation — same for everyone)
        #   • General questions (rules, GPA, graduation, handbook, professors)
        #   • Greetings, help, what can you do

        sem_num = sem_match.group(1) if sem_match else None
        foundation_sem = sem_num in ('1', '2', '3')
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
            # ── Track needed but not yet chosen → ask, then answer ────────
            # Load foundation data if a semester was mentioned so we can
            # still partially answer after they pick a track
            adv_ctx = ctx_ask_track(max_ch)
            if sem_num:
                # Tell Senpai the student asked about a track-dependent semester
                adv_ctx += (
                    f"\n\nNOTE: Student asked about Semester {sem_num}. "
                    f"This semester is track-specific. Once they choose a track, "
                    f"immediately show their semester {sem_num} plan."
                )

        elif sem_num:
            # ── Semester asked — load that semester's data ─────────────────
            if foundation_sem:
                # Foundation semesters: same for everyone, no track needed
                courses, title = load_semester(school or 'ECCE', dept or 'CSE', sem_num)
                if courses:
                    adv_ctx = ctx_semester_plan(
                        courses, title, sem_num,
                        school or 'ECCE', dept or 'CSE', track_label,
                        max_ch, is_half
                    )
                    # If no track yet, append a gentle nudge (not a hard block)
                    if not track_info:
                        adv_ctx += (
                            "\n\n[SOFT NUDGE FOR SENPAI]: After answering, casually ask "
                            "which track the student plans to pursue. Mention that knowing "
                            "their track helps you flag which electives are actually "
                            "prerequisites for their future courses."
                        )
                else:
                    adv_ctx = f"[No data found for Foundation Semester {sem_num}]"

            elif track_info:
                # Track-specific semester with track known
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
                # Track-specific semester but no track → ask
                adv_ctx = ctx_ask_track(max_ch)
                adv_ctx += (
                    f"\n\nNOTE: Student asked about Semester {sem_num} which varies by track. "
                    f"Ask for their track first, then show the semester plan."
                )

        elif track_info:
            # ── Track known, no specific semester → show prereq roadmap ───
            adv_ctx = ctx_track_overview(school, dept, track_label, max_ch, is_half)

        else:
            # ── General question, no semester, no track needed ─────────────
            # Just answer from handbook/general knowledge; no structured data needed
            adv_ctx = (
                "[GENERAL QUESTION — No semester or track-specific data required]\n"
                "Answer the student's question using the handbook context and your general "
                "E-JUST knowledge. You may mention the available tracks and invite them to "
                "share their track if relevant, but do NOT block the answer."
            )

        # ── E. PROFESSOR CONTEXT ──────────────────────────────────────────
        # Detect if user is asking about a professor (by name OR general review query)
        prof_ctx = ""
        asks_prof = any(kw in p_lower for kw in [
            'professor', 'prof', 'doctor', 'dr ', 'dr.', 'instructor',
            'review', 'rating', 'rate', 'recommend a doctor', 'best doctor',
            'who teaches', 'who is teaching', 'avoid', 'good teacher'
        ])

        if profs_df is not None:
            # Try matching a specific name first
            matched_rows = []
            for _, row in profs_df.iterrows():
                pname = str(row.get('Name', '')).lower()
                if any(part in p_lower for part in pname.split() if len(part) > 2):
                    matched_rows.append(row)

            if matched_rows:
                # Specific professor found
                parts = []
                for row in matched_rows:
                    parts.append(
                        f"Professor: {row.get('Name')} | "
                        f"Rating: {row.get('Rating (1-5)', 'N/A')}/5 | "
                        f"Review: {row.get('Review', 'No review available')}"
                    )
                prof_ctx = "\n".join(parts)

            elif asks_prof:
                # General professor query — provide the full list so Senpai can answer
                rows = []
                for _, row in profs_df.iterrows():
                    name   = row.get('Name', 'Unknown')
                    rating = row.get('Rating (1-5)', 'N/A')
                    review = str(row.get('Review', ''))[:120]  # trim long reviews
                    rows.append(f"  • {name} | Rating: {rating}/5 | {review}")
                prof_ctx = (
                    "[ALL PROFESSOR REVIEWS]\n"
                    + "\n".join(rows)
                    + "\n\nSenpai: Use this list to answer questions about ratings, "
                    "recommendations, or comparisons between professors."
                )

        # ── F. DETECT INTENT / MISSION ────────────────────────────────────
        # Tell Senpai which mission is active so it focuses the right capability
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
            "COURSE REGISTRATION ASSISTANCE" if asks_registration else
            "PROFESSOR REVIEWS & RECOMMENDATIONS" if (asks_prof and prof_ctx) else
            "HANDBOOK / ACADEMIC RULES" if asks_handbook else
            "COURSE SCHEDULE & PLANNING" if asks_schedule else
            "GENERAL ADVISING"
        )

        # ── G. SYSTEM PROMPT ──────────────────────────────────────────────
        system_prompt = f"""
You are **Senpai**, the official AI academic advisor for E-JUST (Egypt-Japan University of Science and Technology).
You are friendly, knowledgeable, and direct. Students rely on you for real help — not vague answers.

━━━ YOUR MISSIONS (you handle ALL of these) ━━━
1. 📋 COURSE REGISTRATION HELP
   - Explain how to register/add/drop courses step by step
   - Warn about prerequisites, credit hour limits, and deadlines
   - Help the student build a valid course list for their semester
   - Flag conflicts: if a student wants to take a course they don't have the prereq for, tell them

2. 👨‍🏫 PROFESSOR REVIEWS & RECOMMENDATIONS
   - Share ratings and student reviews from the professor database
   - Recommend professors based on ratings when asked
   - If asked "who is the best doctor for X course", suggest the highest-rated one
   - Be honest but fair — share the data, don't editorialize

3. 📖 HANDBOOK & ACADEMIC RULES
   - Answer questions about university policies, GPA rules, probation, graduation requirements
   - Explain credit hour limits by status (half-load / regular / over-achiever)
   - Answer questions about attendance, exams, appeals, withdrawals, etc.
   - Source answers from the handbook context provided

4. 🗓️ COURSE PLANNING & SCHEDULE ADVICE
   - Help students plan their semester based on their GPA and track
   - Show prerequisite chains so students understand what they need to take first
   - For half-load students: build a tight schedule that fits in 14 CH
   - Flag electives that are secretly prerequisites for later core courses

━━━ ACTIVE MISSION THIS TURN ━━━
{active_mission}

━━━ STUDENT PROFILE ━━━
  • CGPA         : {user_cgpa}
  • Status       : {status_lbl}
  • Credit limit : {max_ch} CH/semester
  • Chosen track : {track_label}
  • Half-Load    : {'YES — Academic Probation (max 14 CH per semester)' if is_half else 'No'}

━━━ CREDIT HOUR RULES (never contradict) ━━━
  • CGPA < 2.0        → Half-Load       — max 14 CH
  • 2.0 ≤ CGPA < 3.0  → Regular Load   — max 19 CH
  • CGPA ≥ 3.0        → Over-Achiever  — max 21 CH

━━━ BEHAVIOUR RULES ━━━
  • ALWAYS answer the question first. Never refuse or deflect.
  • Only ask for the student's track when the answer genuinely requires it (semester 4+, personalised plans).
  • For foundation semesters (1–3), general rules, professor reviews — answer immediately.
  • After answering, you may invite them to share their track to personalise further.
  • Never say "I can only help with course registration" — you help with everything above.
  • Keep responses concise and structured. Use bullet points or tables when it helps.

━━━ STRICT DATA SOURCE RULES ━━━
  • ALL course information (names, codes, credit hours, prerequisites, types) MUST come
    ONLY from the COURSE & SCHEDULE DATA section below (loaded from Tracks.json).
  • NEVER use the handbook to answer questions about courses, semesters, or curricula.
  • NEVER invent, guess, or infer course names or codes that are not in the data.
  • If a course or semester has no data, say clearly: "This data is not available yet."
  • The handbook is ONLY for academic rules, policies, regulations, and procedures.

━━━ COURSE & SCHEDULE DATA ━━━
{adv_ctx}

━━━ HANDBOOK CONTEXT ━━━
{pdf_ctx}

━━━ PROFESSOR DATA ━━━
{prof_ctx if prof_ctx else "No professor data matched this query."}
""".strip()

        # ── G. CALL GEMINI ────────────────────────────────────────────────
        try:
            # Build conversation history in Gemini format
            history = []
            for m in st.session_state.messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})

            chat = gemini_model.start_chat(history=history)
            full_prompt = f"{system_prompt}\n\n{prompt}"
            resp   = chat.send_message(full_prompt)
            answer = resp.text
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

        except Exception as e:
            st.error(f"Gemini API Error: {e}")
