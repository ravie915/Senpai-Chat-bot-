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
# 9. PAGE CONFIG, STYLING & STREAMLIT SESSION STATE
# ════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Senpai — E-JUST Advisor", layout="wide", page_icon="🦉")

# ── Encode logo as base64 ────────────────────────────────────────
import base64
def get_logo_b64():
    logo_paths = ["senpai_logo.jpg", "senpai_logo.png", "logo.jpg", "logo.png"]
    for p in logo_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                ext = p.split(".")[-1]
                mime = "image/jpeg" if ext == "jpg" else "image/png"
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return None

logo_b64 = get_logo_b64()
logo_html = f'<img src="{logo_b64}" class="logo-img" />' if logo_b64 else \
            '<div class="logo-fallback">🦉</div>'

# ── Full custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap');

/* Lemon Milk via CDN */
@font-face {
    font-family: 'LemonMilk';
    src: url('https://cdn.jsdelivr.net/gh/dharmatype/Bebas-Neue@master/fonts/BebasNeue(2019)byDhamraType.otf');
}

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.stApp { background: #ffffff !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none; }

/* Hide default chat elements */
.stChatMessage { display: none !important; }
[data-testid="stChatInput"] { display: none !important; }

/* ── Page layout ── */
.senpai-page {
    min-height: 100vh;
    background: #ffffff;
    font-family: 'DM Sans', sans-serif;
    position: relative;
    overflow-x: hidden;
}

/* ── Red wave top-right ── */
.wave-bg {
    position: fixed;
    top: 0; right: 0;
    width: 52%;
    height: 260px;
    pointer-events: none;
    z-index: 0;
}

/* ── Header ── */
.senpai-header {
    position: relative;
    z-index: 10;
    padding: 28px 52px 16px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-img {
    width: 62px;
    height: 62px;
    object-fit: contain;
}

.logo-fallback {
    font-size: 52px;
    line-height: 1;
}

.brand-text {
    font-family: 'LemonMilk', 'Bebas Neue', 'Impact', sans-serif;
    font-size: 48px;
    letter-spacing: 6px;
    color: #1a1a1a;
    line-height: 1;
    font-weight: 400;
}

/* ── Chat container ── */
.chat-container {
    position: relative;
    z-index: 10;
    padding: 8px 52px 160px;
    display: flex;
    flex-direction: column;
    gap: 18px;
    min-height: 60vh;
}

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px 0 40px;
    gap: 12px;
    opacity: 0.35;
}

.empty-state-owl { font-size: 48px; }
.empty-state-text {
    font-size: 15px;
    color: #666;
    text-align: center;
    max-width: 360px;
    line-height: 1.6;
}

/* ── Message row ── */
.msg-row {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: fadeUp 0.25s ease both;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Avatars ── */
.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    flex-shrink: 0;
    margin-top: 2px;
    letter-spacing: 0.5px;
}
.avatar.bot-av  { background: #1a1a1a; color: #ffffff; font-family: 'LemonMilk','Bebas Neue',sans-serif; font-size: 9px; letter-spacing: 1px; }
.avatar.user-av { background: #c8291a; color: #ffffff; }

/* ── Bubbles ── */
.bubble {
    max-width: 66%;
    padding: 12px 18px;
    font-size: 14.5px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}
.bubble.bot-b {
    background: #f3f2f0;
    color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
}
.bubble.user-b {
    background: #c8291a;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
}

/* ── Input area (fixed bottom) ── */
.input-area {
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 100;
    padding: 12px 52px 24px;
    background: linear-gradient(to top, #ffffff 75%, rgba(255,255,255,0));
}

/* ── Suggestion chips ── */
.chips-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 10px;
}
.chip-btn {
    padding: 6px 16px;
    border-radius: 50px;
    border: 1.5px solid #e0e0e0;
    background: #ffffff;
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px;
    color: #555;
    cursor: pointer;
    transition: all 0.18s;
    white-space: nowrap;
}
.chip-btn:hover {
    border-color: #c8291a;
    color: #c8291a;
    background: #fff5f4;
}

/* ── Input pill ── */
.input-pill {
    display: flex;
    align-items: center;
    background: #ebebeb;
    border: 1.5px solid #d2d2d2;
    border-radius: 50px;
    padding: 8px 8px 8px 22px;
    gap: 8px;
    transition: border-color 0.2s, background 0.2s, box-shadow 0.2s;
}
.input-pill:focus-within {
    border-color: #c8291a;
    background: #ffffff;
    box-shadow: 0 0 0 3px rgba(200,41,26,0.07);
}
.input-pill input {
    flex: 1;
    border: none;
    background: transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #1a1a1a;
    outline: none;
    padding: 4px 0;
}
.input-pill input::placeholder { color: #aaa; }
.send-btn {
    width: 36px; height: 36px;
    border-radius: 50%;
    background: #c8291a;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.18s, transform 0.12s;
    font-size: 14px;
    color: white;
}
.send-btn:hover  { background: #a82215; }
.send-btn:active { transform: scale(0.93); }
</style>
""", unsafe_allow_html=True)

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ── Session state ────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "user_cgpa"  not in st.session_state: st.session_state.user_cgpa  = None
if "track_info" not in st.session_state: st.session_state.track_info = None

# ── Wave SVG ────────────────────────────────────────────────────
wave_svg = """
<svg class="wave-bg" viewBox="0 0 760 260" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g opacity="0.9">
    <path d="M760 0 Q 480 70 180 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 500 80 220 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 520 90 260 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 540 100 300 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 560 110 340 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 580 120 380 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 600 130 420 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 620 140 460 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 640 150 500 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 660 160 540 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 680 170 580 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 700 180 620 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 720 190 660 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M760 0 Q 740 200 700 260" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M460 0 Q 620 75 760 110" stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M500 0 Q 640 70 760 90"  stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M540 0 Q 660 62 760 70"  stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M580 0 Q 680 52 760 52"  stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M620 0 Q 700 42 760 36"  stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M660 0 Q 720 30 760 20"  stroke="#c8291a" stroke-width="0.85" fill="none"/>
    <path d="M700 0 Q 740 16 760 8"   stroke="#c8291a" stroke-width="0.85" fill="none"/>
  </g>
</svg>
"""

# ── Render page shell ────────────────────────────────────────────
st.markdown(f"""
<div class="senpai-page">
  {wave_svg}
  <div class="senpai-header">
    {logo_html}
    <span class="brand-text">SENPAI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ──────────────────────────────────────────
chat_html = '<div class="chat-container">'

if not st.session_state.messages:
    chat_html += """
    <div class="empty-state">
      <div class="empty-state-owl">🦉</div>
      <div class="empty-state-text">Ask me about your courses, professors, schedule, or academic rules</div>
    </div>"""
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"].replace("<", "&lt;").replace(">", "&gt;")
        if role == "assistant":
            chat_html += f"""
            <div class="msg-row bot">
              <div class="avatar bot-av">SP</div>
              <div class="bubble bot-b">{content}</div>
            </div>"""
        else:
            initials = "ME"
            chat_html += f"""
            <div class="msg-row user">
              <div class="avatar user-av">{initials}</div>
              <div class="bubble user-b">{content}</div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# ── Input area with chips + pill ────────────────────────────────
# Show chips only on first message
chips_html = ""
if not st.session_state.messages:
    chips_html = """
    <div class="chips-row">
      <span class="chip-btn">📚 Semester 3 courses</span>
      <span class="chip-btn">👨‍🏫 Best professors</span>
      <span class="chip-btn">📋 How to register</span>
      <span class="chip-btn">📊 My GPA rules</span>
    </div>"""

st.markdown(f"""
<div class="input-area">
  {chips_html}
  <div class="input-pill">
    <span style="font-size:16px">💬</span>
    <input type="text" placeholder="Ask Senpai...." id="senpai-input" disabled />
    <button class="send-btn" disabled>➤</button>
  </div>
</div>
""", unsafe_allow_html=True)

# Actual functional Streamlit input (hidden, provides real interactivity)
st.markdown("""
<style>
div[data-testid="stChatInput"] {
    display: block !important;
    position: fixed;
    bottom: 24px; left: 52px; right: 52px;
    z-index: 200;
    opacity: 0;
    pointer-events: all;
}
div[data-testid="stChatInput"] textarea {
    border-radius: 50px !important;
    background: transparent !important;
    height: 52px !important;
    cursor: text !important;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 10. MAIN CHAT HANDLER
# ════════════════════════════════════════════════════════════════

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    prompt = st.session_state.messages[-1]["content"]

    with st.spinner(""):

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
        track_info  = st.session_state.track_info
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

        sem_num = sem_match.group(1) if sem_match else None
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
        prof_ctx = ""
        asks_prof = any(kw in p_lower for kw in [
            'professor', 'prof', 'doctor', 'dr ', 'dr.', 'instructor',
            'review', 'rating', 'rate', 'recommend a doctor', 'best doctor',
            'who teaches', 'who is teaching', 'avoid', 'good teacher'
        ])
        if profs_df is not None:
            matched_rows = []
            for _, row in profs_df.iterrows():
                pname = str(row.get('Name', '')).lower()
                if any(part in p_lower for part in pname.split() if len(part) > 2):
                    matched_rows.append(row)
            if matched_rows:
                parts = []
                for row in matched_rows:
                    parts.append(
                        f"Professor: {row.get('Name')} | "
                        f"Rating: {row.get('Rating (1-5)', 'N/A')}/5 | "
                        f"Review: {row.get('Review', 'No review available')}"
                    )
                prof_ctx = "\n".join(parts)
            elif asks_prof:
                rows = []
                for _, row in profs_df.iterrows():
                    name   = row.get('Name', 'Unknown')
                    rating = row.get('Rating (1-5)', 'N/A')
                    review = str(row.get('Review', ''))[:120]
                    rows.append(f"  • {name} | Rating: {rating}/5 | {review}")
                prof_ctx = "[ALL PROFESSOR REVIEWS]\n" + "\n".join(rows)

        # ── F. MISSION DETECTION ──────────────────────────────────────────
        asks_registration = any(kw in p_lower for kw in [
            'register', 'registration', 'add course', 'drop course', 'enroll',
            'sign up', 'how to register', 'course registration', 'portal', 'sis'
        ])
        asks_handbook = any(kw in p_lower for kw in [
            'rule', 'policy', 'regulation', 'graduate', 'graduation', 'gpa requirement',
            'academic', 'probation', 'dismissal', 'appeal', 'leave', 'transfer',
            'handbook', 'bylaw', 'attendance', 'exam', 'retake', 'withdraw'
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
You are **Senpai**, the official AI academic advisor for E-JUST.
You are friendly, knowledgeable, and direct.

━━━ YOUR MISSIONS ━━━
1. 📋 COURSE REGISTRATION HELP — explain how to register/add/drop, warn about prereqs and limits
2. 👨‍🏫 PROFESSOR REVIEWS — share ratings and reviews, recommend professors
3. 📖 HANDBOOK & RULES — answer policy questions, GPA rules, graduation requirements
4. 🗓️ COURSE PLANNING — help plan semesters, show prereq chains, build half-load schedules

━━━ ACTIVE MISSION ━━━
{active_mission}

━━━ STUDENT PROFILE ━━━
  • CGPA: {user_cgpa} | Status: {status_lbl} | Limit: {max_ch} CH | Track: {track_label}
  • Half-Load: {'YES — Academic Probation (max 14 CH)' if is_half else 'No'}

━━━ CREDIT RULES ━━━
  • CGPA < 2.0 → Half-Load — max 14 CH
  • 2.0–2.99   → Regular   — max 19 CH
  • ≥ 3.0      → Honors    — max 21 CH

━━━ BEHAVIOUR RULES ━━━
  • ALWAYS answer first. Never refuse.
  • Ask for track only when genuinely needed (sem 4+, personalised plans).
  • Never invent courses — only use data from COURSE & SCHEDULE DATA below.
  • The handbook is ONLY for rules/policies, never for course lists.

━━━ COURSE & SCHEDULE DATA ━━━
{adv_ctx}

━━━ HANDBOOK CONTEXT ━━━
{pdf_ctx}

━━━ PROFESSOR DATA ━━━
{prof_ctx if prof_ctx else "No professor data matched this query."}
""".strip()

        # ── H. CALL GEMINI ────────────────────────────────────────────────
        try:
            history = []
            for m in st.session_state.messages[:-1]:
                role = "user" if m["role"] == "user" else "model"
                history.append({"role": role, "parts": [m["content"]]})
            chat   = gemini_model.start_chat(history=history)
            resp   = chat.send_message(f"{system_prompt}\n\n{prompt}")
            answer = resp.text
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ Gemini API Error: {e}"
            })
            st.rerun()
