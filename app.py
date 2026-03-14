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
    if cgpa < 2.0:
        return "Half-Load (Academic Probation)", 14
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
# 4. COURSE CATALOG
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

# ════════════════════════════════════════════════════════════════
# 5. PREREQUISITE ENGINE
# ════════════════════════════════════════════════════════════════

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

# ════════════════════════════════════════════════════════════════
# 6. SEMESTER DATA LOADER
# ════════════════════════════════════════════════════════════════

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
# 8. CONTEXT BUILDERS
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
# 9. PAGE CONFIG, CUSTOM UI & SESSION STATE
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Senpai — E-JUST Advisor",
    layout="wide",
    page_icon="🦉",
    initial_sidebar_state="collapsed"
)

import base64

def get_image_b64(path: str) -> str | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            ext  = path.rsplit(".", 1)[-1].lower()
            mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png",  "svg": "image/svg+xml"}.get(ext, "image/png")
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    return None

def get_font_b64(path: str) -> str | None:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

# ── Load assets (place these files in repo root) ──────────────────
logo_b64  = get_image_b64("senpai_logo.png") or get_image_b64("senpai_logo.jpg")
wave_b64  = get_image_b64("wave.png") or get_image_b64("wave.svg")
font_b64  = get_font_b64("LemonMilk.otf") or get_font_b64("LemonMilk.woff2") or get_font_b64("LemonMilk.ttf")

# ── Font face ─────────────────────────────────────────────────────
if font_b64:
    font_css = f"""
    @font-face {{
        font-family: 'LemonMilk';
        src: url('data:font/otf;base64,{font_b64}') format('opentype');
        font-weight: normal;
    }}
    """
else:
    # Fallback: Bebas Neue from Google Fonts
    font_css = "@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap');"

brand_font = "'LemonMilk'" if font_b64 else "'Bebas Neue'"

# ── Logo HTML ─────────────────────────────────────────────────────
if logo_b64:
    logo_html = f'<img src="{logo_b64}" class="senpai-logo" />'
else:
    logo_html = '<span class="senpai-logo-fallback">🦉</span>'

# ── Wave HTML ─────────────────────────────────────────────────────
if wave_b64:
    wave_html = f'<img src="{wave_b64}" class="senpai-wave" />'
else:
    # SVG fallback wave
    wave_html = """
    <svg class="senpai-wave" viewBox="0 0 900 340" fill="none" xmlns="http://www.w3.org/2000/svg">
      <g opacity="0.9">
        <path d="M900 0 Q 580 80 200 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 600 90 240 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 620 100 280 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 640 110 320 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 660 120 360 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 680 130 400 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 700 140 440 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 720 150 480 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 740 160 520 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 760 170 560 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 780 180 600 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 800 190 640 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 820 200 680 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 840 215 720 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 860 230 760 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M900 0 Q 875 245 800 340" stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M500 0 Q 680 80 900 130"  stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M540 0 Q 700 72 900 110"  stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M580 0 Q 720 62 900 88"   stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M620 0 Q 740 52 900 66"   stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M660 0 Q 760 40 900 48"   stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M700 0 Q 780 28 900 32"   stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M740 0 Q 800 18 900 18"   stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M780 0 Q 840 10 900 8"    stroke="#c8291a" stroke-width="0.9" fill="none"/>
        <path d="M820 0 Q 860 4  900 2"    stroke="#c8291a" stroke-width="0.9" fill="none"/>
      </g>
    </svg>"""

# ── Full CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
{font_css}

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.stApp {{ background: #ffffff !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
section[data-testid="stSidebar"] {{ display: none; }}
.stChatMessage {{ background: transparent !important; border: none !important; }}

/* ── Wave — full top-right corner, bigger and denser ── */
.senpai-wave {{
    position: fixed;
    top: -10px; right: -10px;
    width: 65%;
    height: 340px;
    object-fit: fill;
    object-position: top right;
    pointer-events: none;
    z-index: 0;
}}

/* ── Header — top-left, compact ── */
.senpai-header {{
    position: fixed;
    top: 0; left: 0;
    z-index: 10;
    padding: 22px 40px;
    display: flex;
    align-items: center;
    gap: 14px;
}}
.senpai-logo {{
    width: 56px;
    height: 56px;
    object-fit: contain;
}}
.senpai-logo-fallback {{
    font-size: 46px;
    line-height: 1;
}}
.senpai-brand {{
    font-family: {brand_font}, 'Bebas Neue', 'Impact', sans-serif;
    font-size: 46px;
    letter-spacing: 6px;
    color: #1a1a1a;
    line-height: 1;
    font-weight: 400;
}}

/* ── Chat area — push down below fixed header ── */
.senpai-chat {{
    position: relative;
    z-index: 5;
    padding: 110px 52px 170px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}}

/* ── Empty state ── */
.senpai-empty {{
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 60px 0 20px;
    gap: 10px;
    opacity: 0.3;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px;
    color: #555;
    text-align: center;
}}

/* ── Message rows ── */
.msg-row {{
    display: flex;
    gap: 12px;
    align-items: flex-start;
    animation: fadeUp 0.2s ease both;
}}
.msg-row.user {{ flex-direction: row-reverse; }}
@keyframes fadeUp {{
    from {{ opacity: 0; transform: translateY(6px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}

/* ── Avatars ── */
.msg-avatar {{
    width: 34px; height: 34px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 600;
    flex-shrink: 0; margin-top: 2px;
    letter-spacing: 0.5px;
}}
.msg-avatar.bot  {{
    background: #1a1a1a; color: #fff;
    font-family: {brand_font}, sans-serif;
    font-size: 8px; letter-spacing: 1px;
}}
.msg-avatar.user {{ background: #c8291a; color: #fff; font-size: 11px; }}

/* ── Bubbles ── */
.msg-bubble {{
    max-width: 68%;
    padding: 12px 18px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14.5px;
    line-height: 1.65;
    white-space: pre-wrap;
    word-break: break-word;
}}
.msg-bubble.bot  {{
    background: #f3f2f0;
    color: #1a1a1a;
    border-radius: 18px 18px 18px 4px;
}}
.msg-bubble.user {{
    background: #c8291a;
    color: #ffffff;
    border-radius: 18px 18px 4px 18px;
}}

/* ── Fixed bottom input area — full width, white bg ── */
.senpai-input-area {{
    position: fixed;
    bottom: 0; left: 0; right: 0;
    z-index: 100;
    padding: 16px 40px 28px;
    background: linear-gradient(to top, #ffffff 75%, rgba(255,255,255,0));
}}

/* ── Chips ── */
.senpai-chips {{
    display: flex; gap: 8px; flex-wrap: wrap;
    margin-bottom: 10px;
}}
.senpai-chip {{
    padding: 6px 16px;
    border-radius: 50px;
    border: 1.5px solid #e0e0e0;
    background: #fff;
    font-family: 'DM Sans', sans-serif;
    font-size: 12.5px; color: #555;
    cursor: default;
    white-space: nowrap;
}}

/* ── Input pill — full width, white, red border on focus ── */
.senpai-pill {{
    display: flex; align-items: center;
    background: #ffffff;
    border: 2px solid #d2d2d2;
    border-radius: 50px;
    padding: 10px 10px 10px 24px;
    gap: 10px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.senpai-pill:focus-within {{
    border-color: #c8291a;
    box-shadow: 0 0 0 3px rgba(200,41,26,0.08);
}}
.senpai-pill input {{
    flex: 1; border: none; background: transparent;
    font-family: 'DM Sans', sans-serif;
    font-size: 15px; color: #1a1a1a; outline: none;
    padding: 4px 0;
}}
.senpai-pill input::placeholder {{ color: #aaa; }}
.senpai-send {{
    width: 40px; height: 40px; border-radius: 50%;
    background: #c8291a; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 16px; flex-shrink: 0;
    transition: background 0.18s, transform 0.12s;
}}
.senpai-send:hover {{ background: #a82215; }}

/* ── Make real st.chat_input overlay the pill — full width ── */
div[data-testid="stChatInput"] {{
    position: fixed !important;
    bottom: 28px !important;
    left: 40px !important;
    right: 40px !important;
    z-index: 200 !important;
    opacity: 0.01 !important;
    pointer-events: all !important;
}}
div[data-testid="stChatInput"] textarea {{
    border-radius: 50px !important;
    height: 56px !important;
}}
</style>
""", unsafe_allow_html=True)

# ── API Client ────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Session state ─────────────────────────────────────────────────
if "messages"   not in st.session_state: st.session_state.messages   = []
if "user_cgpa"  not in st.session_state: st.session_state.user_cgpa  = None
if "track_info" not in st.session_state: st.session_state.track_info = None

# ── Render header ─────────────────────────────────────────────────
st.markdown(f"""
<div>
  {wave_html}
  <div class="senpai-header">
    {logo_html}
    <span class="senpai-brand">SENPAI</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Render chat history ───────────────────────────────────────────
chat_html = '<div class="senpai-chat">'
if not st.session_state.messages:
    chat_html += """
    <div class="senpai-empty">
      <div style="font-size:40px">🦉</div>
      <div>Ask me about courses, professors, schedules, or academic rules</div>
    </div>"""
else:
    for msg in st.session_state.messages:
        role    = msg["role"]
        content = msg["content"].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        if role == "assistant":
            chat_html += f"""
            <div class="msg-row bot">
              <div class="msg-avatar bot">SP</div>
              <div class="msg-bubble bot">{content}</div>
            </div>"""
        else:
            chat_html += f"""
            <div class="msg-row user">
              <div class="msg-avatar user">ME</div>
              <div class="msg-bubble user">{content}</div>
            </div>"""
chat_html += "</div>"
st.markdown(chat_html, unsafe_allow_html=True)

# ── Render input area (decorative — real input is st.chat_input) ──
chips_html = ""
if not st.session_state.messages:
    chips_html = """
    <div class="senpai-chips">
      <span class="senpai-chip">📚 Semester 3 courses</span>
      <span class="senpai-chip">👨‍🏫 Find a professor</span>
      <span class="senpai-chip">📋 How to register</span>
      <span class="senpai-chip">📊 GPA rules</span>
    </div>"""

st.markdown(f"""
<div class="senpai-input-area">
  {chips_html}
  <div class="senpai-pill">
    <span style="font-size:15px;color:#aaa">💬</span>
    <input type="text" placeholder="Ask Senpai...." disabled />
    <button class="senpai-send">➤</button>
  </div>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 10. MAIN CHAT HANDLER
# ════════════════════════════════════════════════════════════════

if prompt := st.chat_input("Ask Senpai...."):
    st.session_state.messages.append({"role": "user", "content": prompt})

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
        matched_rows = []
        query_words = [w for w in p_lower.split() if len(w) > 2]

        scored = []
        for _, row in profs_df.iterrows():
            pname = str(row.get('Name', '')).lower()
            pname_parts = [p for p in pname.split() if len(p) > 2]
            score = 0
            score += sum(1 for qw in query_words if qw in pname) * 2
            score += sum(1 for pp in pname_parts if pp in p_lower)
            if score > 0:
                scored.append((score, row))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            top_score = scored[0][0]
            matched_rows = [row for s, row in scored if s >= top_score - 1]
            matched_rows = matched_rows[:5]

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
  • Half-Load: {'YES — Academic Probation (max 14 CH)' if is_half else 'No'}

━━━ CREDIT RULES ━━━
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
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    except Exception as e:
        st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {e}"})
        st.rerun()
