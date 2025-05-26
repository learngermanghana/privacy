import re
import csv
import os
import streamlit as st
import pandas as pd
import openai
import stopwordsiso as stop
from datetime import datetime

# === Paths & Constants ===
TRAINING_DATA_PATH   = "essay_training_data.csv"
VOCAB_PATH           = "approved_vocab.csv"
CONNECTOR_PATH       = "approved_connectors.csv"
LOG_PATH             = "submission_log.csv"
STUDENT_CODES_PATH   = "student_codes.csv"

st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")
st.title("📝 German Letter & Essay Checker – Learn Language Education Academy")
st.markdown("""
    <style>
    #MainMenu, footer, header, div[data-testid="stHeader"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# === API KEY ===
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as environment variable.")
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# === Stopwords & Function Words ===
german_stopwords = stop.stopwords("de")
function_words = {
    w.lower() for w in {
        "ich","du","er","sie","wir","ihr","mir","mich",
        "der","die","das","ein","eine",
        "und","oder","aber","nicht","wie","wann","wo",
        "sein","haben","können","müssen","wollen","sollen",
        "bitte","viel","gut","sehr",
        "wer","was","wann","wo","warum","wie","order"
    }
}

# === Helpers: File/Data ===
def load_student_codes():
    codes = set()
    if os.path.exists(STUDENT_CODES_PATH):
        with open(STUDENT_CODES_PATH, newline='', encoding='utf-8') as f:
            rdr = csv.reader(f)
            headers = next(rdr, [])
            idx = headers.index('student_code') if 'student_code' in headers else 0
            for row in rdr:
                if len(row) > idx and row[idx].strip():
                    codes.add(row[idx].strip())
    return codes

def load_submission_log():
    data = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                sid, count = row[0], row[1]
                if sid.lower() == 'student_code':
                    continue
                try:
                    data[sid] = int(count)
                except ValueError:
                    continue
    return data

def save_submission_log(log: dict):
    with open(LOG_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for sid, count in log.items():
            writer.writerow([sid, count])

def save_for_training(student_id, level, task_type, task_num, student_text, gpt_results, feedback_text):
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "student_id": student_id,
        "level": level,
        "task_type": task_type,
        "task_num": task_num,
        "original_text": student_text,
        "gpt_grammar_feedback": "\n".join(gpt_results),
        "full_feedback": feedback_text
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRAINING_DATA_PATH) or os.stat(TRAINING_DATA_PATH).st_size == 0:
        df.to_csv(TRAINING_DATA_PATH, index=False)
    else:
        df.to_csv(TRAINING_DATA_PATH, mode='a', header=False, index=False)

def load_vocab_from_csv():
    vocab = {"A1": set(), "A2": set()}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, newline='', encoding='utf-8') as f:
            for lvl, word in csv.reader(f):
                if lvl in vocab:
                    vocab[lvl].add(word.strip().lower())
    if not any(vocab.values()):
        vocab["A1"].update(["bitte","danke","tschüss","möchten","schreiben","bezahlen"])
        vocab["A2"].update(["arbeiten","lernen","verstehen","helfen"])
    return vocab

def save_vocab_to_csv(vocab):
    with open(VOCAB_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for lvl in vocab:
            for w in sorted(vocab[lvl]):
                writer.writerow([lvl, w])

def load_connectors_from_csv():
    conns = {l:set() for l in ["A1","A2","B1","B2"]}
    if os.path.exists(CONNECTOR_PATH):
        with open(CONNECTOR_PATH, newline='', encoding="utf-8") as f:
            for lvl, items in csv.reader(f):
                if lvl in conns:
                    conns[lvl].update(i.strip() for i in items.split(",") if i.strip())
    if not any(conns.values()):
        conns = {
            "A1": {"weil","denn","ich möchte wissen","deshalb"},
            "A2": {"deshalb","deswegen","darum","trotzdem","obwohl","sobald",
                   "außerdem","zum Beispiel","und","aber","oder","erstens",
                   "zweitens","zum Schluss"},
            "B1": {"jedoch","allerdings","hingegen","dennoch","folglich","daher",
                   "deshalb","damit","sofern","falls","währenddessen","anschließend",
                   "einerseits","andererseits"},
            "B2": {"allerdings","dennoch","demzufolge","ergo","sodass","obgleich",
                   "ungeachtet","indessen","nichtsdestotrotz","inzwischen",
                   "zusammenfassend","abschließend","letztendlich"}
        }
    return conns

def save_connectors_to_csv(conns):
    with open(CONNECTOR_PATH, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for lvl in conns:
            if conns[lvl]:
                writer.writerow([lvl, ", ".join(sorted(conns[lvl]))])

# === Long Phrase Detection ===
def detect_long_phrases(text: str, level: str) -> list[str]:
    thresh = 6 if level == "A1" else 8 if level == "A2" else 1000  # Only for A1/A2
    if level not in ("A1","A2"):
        return []
    tokens = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\b\w+\b", text)]
    phrases = []
    for i in range(len(tokens) - thresh + 1):
        start_idx = tokens[i][1]
        end_idx   = tokens[i + thresh - 1][2]
        phrase = text[start_idx:end_idx]
        if len(re.findall(r"\b\w+\b", phrase)) >= thresh:
            phrases.append(phrase)
    # Remove overlaps: only highlight longest unique spans
    seen = set()
    unique = []
    for ph in phrases:
        if ph not in seen:
            seen.add(ph)
            unique.append(ph)
    return unique

# === GPT Grammar Check ===
def grammar_check_with_gpt(text: str) -> list[str]:
    prompt = (
        "You are a German language tutor. "
        "Check the following German text for grammar and spelling errors. "
        "For each error, return a line in this format:\n"
        "`<error substring>` ⇒ `<suggestion>` — `<brief English explanation>`\n\n"
        f"Text:\n{text}"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().splitlines()

# === Annotate Text (Grammar, Long Phrases) ===
def annotate_text(student_text, gpt_results, long_phrases, level):
    ann = student_text
    colors = {'Grammar':'#e15759','Phrase':'#f1c232'}
    # Highlight grammar errors
    for line in gpt_results or []:
        if "⇒" in line:
            err = line.split("⇒")[0].strip(" `")
            pat = rf"(?i)\b{re.escape(err)}\b"
            ann = re.sub(pat,
                         lambda m: f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>",
                         ann)
    # Highlight long phrases for A1/A2
    for phrase in long_phrases:
        pat = re.escape(phrase)
        ann = re.sub(pat,
                     lambda m: f"<span title='Too long for {level}' style='background-color:{colors['Phrase']}; color:#000'>{m.group(0)}</span>",
                     ann)
    return ann.replace("\n", "  \n")

# === Scoring & Feedback ===
def score_text(student_text, gpt_results):
    words = re.findall(r"\w+", student_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    sentences = re.split(r"[.!?]", student_text)
    avg_words = len(words) / max(1, len([s for s in sentences if s.strip()]))
    readability = "Easy" if avg_words <= 12 else "Medium" if avg_words <= 17 else "Hard"
    content_score   = 10
    grammar_score   = max(1, 5 - len(gpt_results))
    vocab_score     = min(5, int(unique_ratio * 5))
    structure_score = 5
    total           = content_score + grammar_score + vocab_score + structure_score
    return (content_score, grammar_score, vocab_score,
            structure_score, total, unique_ratio, avg_words, readability)

def generate_feedback_text(level, task_type, task, content_score, grammar_score, vocab_score,
                          structure_score, total, gpt_results, long_phrases, student_text):
    return f"""Your Feedback – {task_type} ({level})
Task: {task['task'] if task else ''}
Scores:
- Content: {content_score}/10
- Grammar: {grammar_score}/5
- Vocabulary: {vocab_score}/5
- Structure: {structure_score}/5
Total: {total}/25

Grammar Suggestions:
{chr(10).join(gpt_results) if gpt_results else 'No major grammar errors detected.'}

Long Phrases (flagged as too long):
{', '.join(long_phrases) if long_phrases else 'None'}

Your Text:
{student_text}
"""

# === A1 Schreiben Tasks ===
a1_tasks = {
    1: {"task": "Schreiben Sie eine E-Mail an Ihren Arzt und sagen Sie Ihren Termin ab.",
        "points": ["Warum schreiben Sie?", "Sagen Sie: den Grund für die Absage.", "Fragen Sie: nach einem neuen Termin."]},
    2: {"task": "Schreiben Sie eine Einladung an Ihren Freund zur Feier Ihres neuen Jobs.",
        "points": ["Warum schreiben Sie?", "Wann ist die Feier?", "Wer soll was mitbringen?"]},
    3: {"task": "Schreiben Sie eine E-Mail an einen Freund und teilen Sie ihm mit, dass Sie ihn besuchen möchten.",
        "points": ["Warum schreiben Sie?", "Wann besuchen Sie ihn?", "Was möchten Sie zusammen machen?"]},
    4: {"task": "Schreiben Sie eine E-Mail an Ihre Schule und fragen Sie nach einem Deutschkurs.",
        "points": ["Warum schreiben Sie?", "Was möchten Sie wissen?", "Wie kann die Schule antworten?"]},
    5: {"task": "Schreiben Sie eine E-Mail an Ihre Vermieterin. Ihre Heizung ist kaputt.",
        "points": ["Warum schreiben Sie?", "Seit wann ist die Heizung kaputt?", "Was soll die Vermieterin tun?"]},
    6: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben eine neue Wohnung.",
        "points": ["Warum schreiben Sie?", "Wo ist die Wohnung?", "Was gefällt Ihnen?"]},
    7: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben eine neue Arbeitsstelle.",
        "points": ["Warum schreiben Sie?", "Wo arbeiten Sie jetzt?", "Was machen Sie?"]},
    8: {"task": "Schreiben Sie eine E-Mail an Ihren Lehrer. Sie können am Kurs nicht teilnehmen.",
        "points": ["Warum schreiben Sie?", "Warum kommen Sie nicht?", "Was möchten Sie?"]},
    9: {"task": "Schreiben Sie eine E-Mail an die Bibliothek. Sie haben ein Buch verloren.",
        "points": ["Warum schreiben Sie?", "Welches Buch haben Sie verloren?", "Was möchten Sie wissen?"]},
    10: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten mit ihr in den Urlaub fahren.",
         "points": ["Warum schreiben Sie?", "Wohin möchten Sie fahren?", "Was möchten Sie dort machen?"]},
    11: {"task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Termin ändern.",
         "points": ["Warum schreiben Sie?", "Welcher Termin ist es?", "Wann haben Sie Zeit?"]},
    12: {"task": "Schreiben Sie eine E-Mail an Ihren Bruder. Sie machen eine Party.",
         "points": ["Warum schreiben Sie?", "Wann ist die Party?", "Was soll Ihr Bruder mitbringen?"]},
    13: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie sind krank.",
         "points": ["Warum schreiben Sie?", "Was machen Sie heute nicht?", "Was sollen Sie tun?"]},
    14: {"task": "Schreiben Sie eine E-Mail an Ihre Nachbarn. Sie machen Urlaub.",
         "points": ["Warum schreiben Sie?", "Wie lange sind Sie weg?", "Was sollen die Nachbarn tun?"]},
    15: {"task": "Schreiben Sie eine E-Mail an Ihre Deutschlehrerin. Sie möchten eine Prüfung machen.",
         "points": ["Warum schreiben Sie?", "Welche Prüfung möchten Sie machen?", "Wann möchten Sie die Prüfung machen?"]},
    16: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben einen neuen Computer gekauft.",
         "points": ["Warum schreiben Sie?", "Wo haben Sie den Computer gekauft?", "Was gefällt Ihnen besonders?"]},
    17: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen Sport machen.",
         "points": ["Warum schreiben Sie?", "Welchen Sport möchten Sie machen?", "Wann können Sie?"]},
    18: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie brauchen Hilfe beim Umzug.",
         "points": ["Warum schreiben Sie?", "Wann ist der Umzug?", "Was soll Ihr Freund machen?"]},
    19: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten ein Fest organisieren.",
         "points": ["Warum schreiben Sie?", "Wo soll das Fest sein?", "Was möchten Sie machen?"]},
    20: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen kochen.",
         "points": ["Warum schreiben Sie?", "Was möchten Sie kochen?", "Wann möchten Sie kochen?"]},
    21: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben einen neuen Job.",
         "points": ["Warum schreiben Sie?", "Wo arbeiten Sie jetzt?", "Was machen Sie?"]},
    22: {"task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Deutschkurs besuchen.",
         "points": ["Warum schreiben Sie?", "Wann möchten Sie den Kurs besuchen?", "Was möchten Sie noch wissen?"]}
}

# === Teacher Settings ===
st.sidebar.header("🔧 Teacher Settings")
teacher_password = st.sidebar.text_input("🔒 Enter teacher password", type="password")
teacher_mode = (teacher_password == "Felix029")
if teacher_mode:
    page = st.sidebar.radio("Go to:", ["Student View", "Teacher Dashboard"])
else:
    page = "Student View"

def download_training_data():
    if os.path.exists(TRAINING_DATA_PATH) and os.stat(TRAINING_DATA_PATH).st_size > 0:
        with open(TRAINING_DATA_PATH, 'rb') as f:
            st.download_button(
                "⬇️ Download All Submissions",
                data=f,
                file_name="essay_training_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No training data collected yet.")

def render_collected_essays_for_training():
    st.subheader("📊 Collected Essays for AI Training")
    download_training_data()

# === Teacher Dashboard ===
if teacher_mode and page == "Teacher Dashboard":
    st.header("📋 Teacher Dashboard")

    # ---- VOCABULARY EDITOR ----
    st.subheader("Edit Approved Vocabulary (A1/A2)")
    vocab = load_vocab_from_csv()
    for lvl in ["A1", "A2"]:
        current = ", ".join(sorted(vocab.get(lvl, set())))
        new_vocab = st.text_area(f"{lvl} Vocabulary (comma-separated):", current, key=f"vocab_{lvl}")
        if st.button(f"Update {lvl} Vocab", key=f"btn_vocab_{lvl}"):
            words = {w.strip().lower() for w in new_vocab.split(",") if w.strip()}
            vocab[lvl] = words
            save_vocab_to_csv(vocab)
            st.success(f"✅ {lvl} vocabulary updated.")

    # ---- CONNECTORS EDITOR ----
    st.subheader("Edit Approved Connectors (A1-B2)")
    connectors = load_connectors_from_csv()
    for lvl in ["A1", "A2", "B1", "B2"]:
        current = ", ".join(sorted(connectors.get(lvl, set())))
        new_conns = st.text_area(f"{lvl} Connectors (comma-separated):", current, key=f"conn_{lvl}")
        if st.button(f"Update {lvl} Connectors", key=f"btn_conn_{lvl}"):
            items = {c.strip() for c in new_con
# --- Teacher Settings ---
st.sidebar.header("🔧 Teacher Settings")
teacher_password = st.sidebar.text_input(
    "🔒 Enter teacher password",
    type="password",
    key="admin123"  
)
teacher_mode = (teacher_password == "Felix029")
if teacher_mode:
    page = st.sidebar.radio(
        "Go to:",
        ["Student View", "Teacher Dashboard"],
        key="Moxflex029"     
    )
else:
    page = "Student View"

def download_training_data():
    """Offer the training data CSV as a download in the dashboard."""
    if os.path.exists(TRAINING_DATA_PATH) and os.stat(TRAINING_DATA_PATH).st_size > 0:
        with open(TRAINING_DATA_PATH, 'rb') as f:
            st.download_button(
                "⬇️ Download All Submissions",
                data=f,
                file_name="essay_training_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No training data collected yet.")

if teacher_mode and page == "Teacher Dashboard":
    st.header("📋 Teacher Dashboard")

    with st.expander("📝 Edit Approved Vocabulary (A1/A2)"):
        vocab = load_vocab_from_csv()
        for lvl in ["A1", "A2"]:
            current = ", ".join(sorted(vocab.get(lvl, set())))
            new_vocab = st.text_area(f"{lvl} Vocabulary (comma-separated):", current, key=f"vocab_{lvl}")
            if st.button(f"Update {lvl} Vocab", key=f"btn_vocab_{lvl}"):
                words = {w.strip().lower() for w in new_vocab.split(",") if w.strip()}
                vocab[lvl] = words
                save_vocab_to_csv(vocab)
                st.success(f"✅ {lvl} vocabulary updated.")

    with st.expander("🔗 Edit Approved Connectors (A1–B2)"):
        connectors = load_connectors_from_csv()
        for lvl in ["A1", "A2", "B1", "B2"]:
            current = ", ".join(sorted(connectors.get(lvl, set())))
            new_conns = st.text_area(f"{lvl} Connectors (comma-separated):", current, key=f"conn_{lvl}")
            if st.button(f"Update {lvl} Connectors", key=f"btn_conn_{lvl}"):
                items = {c.strip() for c in new_conns.split(",") if c.strip()}
                connectors[lvl] = items
                save_connectors_to_csv(connectors)
                st.success(f"✅ {lvl} connectors updated.")

    with st.expander("👩‍🎓 Student Codes"):
        student_codes = load_student_codes()
        st.write(sorted(student_codes))
        new_codes = st.text_area("Add student codes (comma-separated):")
        if st.button("Add to Student Codes"):
            for code in [s.strip() for s in new_codes.split(',') if s.strip()]:
                student_codes.add(code)
            with open(STUDENT_CODES_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["student_code"])
                for c in sorted(student_codes):
                    writer.writerow([c])
            st.success("✅ Student codes updated.")

    with st.expander("📈 Submission Log"):
        df_log = pd.DataFrame(load_submission_log().items(), columns=["student_code","count"])
        st.dataframe(df_log)
        st.download_button(
            "💾 Download submission_log.csv",
            data=df_log.to_csv(index=False).encode('utf-8'),
            file_name="submission_log.csv",
            mime="text/csv"
        )
        uploaded = st.file_uploader(
            "🔄 Upload previous submission_log.csv to restore counts",
            type="csv",
            help="After deploying new code, re-import your last export here."
        )
        if uploaded:
            with open(LOG_PATH, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("✅ Submission log restored. Reload the page to see updated counts.")

    with st.expander("📊 Collected Essays for AI Training"):
        download_training_data()

    st.stop()

# --- Scoring Helper ---
def score_text(student_text, level, gpt_results, adv):
    words = re.findall(r"\w+", student_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    sentences = re.split(r"[.!?]", student_text)
    avg_words = len(words) / max(1, len([s for s in sentences if s.strip()]))
    readability = "Easy" if avg_words <= 12 else "Medium" if avg_words <= 17 else "Hard"
    content_score   = 10
    grammar_score   = max(1, 5 - len(gpt_results))
    vocab_score     = min(5, int((len(set(words)) / len(words)) * 5))
    if adv:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total           = content_score + grammar_score + vocab_score + structure_score
    return (content_score, grammar_score, vocab_score,
            structure_score, total, unique_ratio, avg_words, readability)

# --- Why these scores and feedback ---
def explain_scores(content_score, grammar_score, vocab_score, structure_score, unique_ratio, gpt_results):
    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: unique ratio {unique_ratio:.2f} ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

# --- Annotate all the colors ---
def annotate_text(student_text, gpt_results, long_phrases, connectors, level):
    ann = student_text
    colors = {
        'Grammar':   '#e15759',   # 🔴 Red
        'LongPhrase':'#f1c232',   # 🟡 Yellow
        'Connector': '#6aa84f',   # 🟢 Green
        'Passive':   '#e69138',   # 🟠 Orange
        'LongSent':  '#cccccc',   # ⚪️ Gray
        'Noun':      '#e69138',   # 🟠 Orange
        'Repeat':    '#e15759',   # 🔴 Red underline
    }
    # 1) Grammar errors (red)
    for line in gpt_results or []:
        if "⇒" in line:
            err = line.split("⇒")[0].strip(" `")
            pat = rf"(?i)\b{re.escape(err)}\b"
            ann = re.sub(pat,
                         lambda m: f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>",
                         ann)
    # 2) Long phrase detection (yellow)
    for phrase in long_phrases:
        pat = re.escape(phrase)
        ann = re.sub(pat,
                     lambda m: f"<span title='Too long for {level}' style='background-color:{colors['LongPhrase']}; color:#000'>{m.group(0)}</span>",
                     ann)
    # 3) Connectors (green)
    for conn in connectors.get(level, []):
        pat = rf"(?i)\b{re.escape(conn)}\b"
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Connector']}; color:#fff'>{m.group(0)}</span>",
                     ann)
    # 4) Passive constructions (orange background)
    for pat in [r"\bwird\s+\w+\s+von\b", r"\bist\s+\w+\s+worden\b"]:
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Passive']}; color:#000'>{m.group(0)}</span>",
                     ann, flags=re.I)
    # 5) Long sentences (gray)
    ann = re.sub(r"([A-ZÄÖÜ][^\.!?]{100,}[\.!?])",
                 lambda m: f"<span style='background-color:{colors['LongSent']}; color:#000'>{m.group(0)}</span>",
                 ann)
    # 6) Noun capitalization issues (orange background)
    for det in [' der',' die',' das',' ein',' eine',' mein',' dein']:
        pat = rf"(?<={det}\s)([a-zäöüß]+)\b"
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Noun']}; color:#fff'>{m.group(0)}</span>",
                     ann)
    # 7) Double spaces (red border)
    ann = re.sub(r" {2,}",
                 lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
                 ann)
    # 8) Missing space after comma (red border)
    ann = re.sub(r",(?=[A-Za-zÖÜÄ])",
                 lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
                 ann)
    # 9) Repeated words (underline)
    ann = re.sub(r"\b(\w+)\s+\1\b",
                 lambda m: f"<span style='text-decoration:underline; color:{colors['Repeat']}'>{m.group(0)}</span>",
                 ann, flags=re.I)
    return ann.replace("\n", "  \n")
