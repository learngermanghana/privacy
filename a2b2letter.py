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
def score_text(student_text, level, gpt_results, adv):
    words = re.findall(r"\w+", student_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    sentences = re.split(r"[.!?]", student_text)
    avg_words = len(words) / max(1, len([s for s in sentences if s.strip()]))
    if avg_words <= 12:
        readability = "Easy"
    elif avg_words <= 17:
        readability = "Medium"
    else:
        readability = "Hard"
    content_score   = 10
    grammar_score   = max(1, 5 - len(gpt_results))
    vocab_score     = min(5, int((len(set(words)) / len(words)) * 5))
    if adv:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total           = content_score + grammar_score + vocab_score + structure_score
    return (content_score, grammar_score, vocab_score,
            structure_score, total, unique_ratio, avg_words, readability)

def generate_feedback_text(level, task_type, task,
                          content_score, grammar_score, vocab_score,
                          structure_score, total,
                          gpt_results, long_phrases, used_connectors, student_text):
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

Connectors Used:
{', '.join(used_connectors) if used_connectors else 'None'}

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

# --- Teacher Settings ---
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

if teacher_mode and page == "Teacher Dashboard":
    st.header("📋 Teacher Dashboard")

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
        student_codes = sorted(load_student_codes())
        st.write("**Current student codes:**")
        st.write(student_codes)

        new_codes = st.text_area("Add student codes (comma-separated):", "")
        if st.button("Add to Student Codes"):
            code_set = set(student_codes)
            for code in [s.strip() for s in new_codes.split(',') if s.strip()]:
                code_set.add(code)
            with open(STUDENT_CODES_PATH, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["student_code"])
                for c in sorted(code_set):
                    writer.writerow([c])
            st.success("✅ Student codes updated.")
            student_codes = sorted(load_student_codes())
            st.write(student_codes)

        # Remove a code
        if student_codes:
            code_to_remove = st.selectbox("Remove a student code:", student_codes)
            if st.button("Remove Selected Code"):
                code_set = set(student_codes)
                code_set.discard(code_to_remove)
                with open(STUDENT_CODES_PATH, "w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["student_code"])
                    for c in sorted(code_set):
                        writer.writerow([c])
                st.success(f"✅ Student code '{code_to_remove}' removed.")
                student_codes = sorted(load_student_codes())
                st.write(student_codes)
        else:
            st.info("No student codes yet.")

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
            type="csv"
        )
        if uploaded:
            with open(LOG_PATH, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success("✅ Submission log restored. Reload the page to see updated counts.")

    with st.expander("📊 Collected Essays for AI Training"):
        download_training_data()

    st.stop()

# --- Annotate Text Helper ---
def annotate_text(student_text, gpt_results, long_phrases, connectors_by_level, level):
    ann = student_text
    colors = {
        'Grammar':   '#e15759',
        'Phrase':    '#f1c232',
        'Connector': '#6aa84f',
        'Passive':   '#e69138',
        'LongSent':  '#cccccc',
        'Noun':      '#e69138',
        'Repeat':    '#e15759'
    }
    for line in gpt_results or []:
        if "⇒" in line:
            err = line.split("⇒")[0].strip(" `")
            pat = rf"(?i)\b{re.escape(err)}\b"
            ann = re.sub(
                pat,
                lambda m: f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>",
                ann
            )
    for phrase in long_phrases:
        pat = re.escape(phrase)
        ann = re.sub(
            pat,
            lambda m: f"<span title='Too long for {level}' style='background-color:{colors['Phrase']}; color:#000'>{m.group(0)}</span>",
            ann
        )
    for conn in connectors_by_level.get(level, []):
        pat = rf"(?i)\b{re.escape(conn)}\b"
        ann = re.sub(
            pat,
            lambda m: f"<span style='background-color:{colors['Connector']}; color:#fff'>{m.group(0)}</span>",
            ann
        )
    for pat in [r"\bwird\s+\w+\s+von\b", r"\bist\s+\w+\s+worden\b"]:
        ann = re.sub(
            pat,
            lambda m: f"<span style='background-color:{colors['Passive']}; color:#000'>{m.group(0)}</span>",
            ann, flags=re.I
        )
    ann = re.sub(
        r"([A-ZÄÖÜ][^\.!?]{100,}[\.!?])",
        lambda m: f"<span style='background-color:{colors['LongSent']}; color:#000'>{m.group(0)}</span>",
        ann
    )
    for det in [' der',' die',' das',' ein',' eine',' mein',' dein']:
        pat = rf"(?<={det}\s)([a-zäöüß]+)\b"
        ann = re.sub(
            pat,
            lambda m: f"<span style='background-color:{colors['Noun']}; color:#fff'>{m.group(0)}</span>",
            ann
        )
    ann = re.sub(
        r" {2,}",
        lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
        ann
    )
    ann = re.sub(
        r",(?=[A-Za-zÖÜÄ])",
        lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
        ann
    )
    ann = re.sub(
        r"\b(\w+)\s+\1\b",
        lambda m: f"<span style='text-decoration:underline; color:{colors['Repeat']}'>{m.group(0)}</span>",
        ann, flags=re.I
    )
    return ann.replace("\n", "  \n")

# --- Student Interface & Feedback ---
approved_vocab      = load_vocab_from_csv()
student_codes       = load_student_codes()
log_data            = load_submission_log()
connectors_by_level = load_connectors_from_csv()

level = st.selectbox("Select your level", ["A1","A2","B1","B2"])
tasks = ["Formal Letter","Informal Letter"]
if level in ("B1","B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select task type", tasks)

student_id = st.text_input("Enter your student code:")
if not student_id:
    st.warning("Please enter your student code.")
    st.stop()
if student_id not in student_codes:
    st.error("❌ You are not authorized to use this app.")
    st.stop()

subs     = log_data.get(student_id, 0)
max_subs = 40 if level == "A1" else 45
if subs >= max_subs:
    st.warning(f"⚠️ You have reached the maximum of {max_subs} submissions.")
    st.stop()
if subs >= max_subs - 12:
    st.info("⏳ You have used most of your submission chances. Review carefully!")

if level == "A1":
    task_num = st.number_input(
        f"Choose a Schreiben task number (1–{len(a1_tasks)})",
        1, len(a1_tasks), 1
    )
    task = a1_tasks.get(task_num)
    st.markdown(f"### Aufgabe {task_num}: {task['task']}")
    st.markdown("**Points:**")
    for p in task["points"]:
        st.markdown(f"- {p}")
else:
    task_num = None
    task = None

student_text = st.text_area("✏️ Write your letter or essay below:", height=300)

if st.button("✅ Submit for Feedback"):
    with st.spinner("🔄 Processing your submission…"):
        if not student_text.strip():
            st.warning("Please enter your text before submitting.")
            st.stop()

        gpt_results = grammar_check_with_gpt(student_text)
        adv = detect_long_phrases(student_text, level)
        used_connectors = [
            c for c in connectors_by_level.get(level, [])
            if c.lower() in student_text.lower()
        ]
        (content_score, grammar_score, vocab_score,
         structure_score, total, unique_ratio, avg_words,
         readability) = score_text(student_text, level, gpt_results, adv)
        feedback_text = generate_feedback_text(
            level, task_type, task, content_score, grammar_score,
            vocab_score, structure_score, total,
            gpt_results, adv, used_connectors, student_text
        )
        save_for_training(
            student_id=student_id,
            level=level,
            task_type=task_type,
            task_num=task_num,
            student_text=student_text,
            gpt_results=gpt_results,
            feedback_text=feedback_text
        )
        log_data[student_id] = subs + 1
        save_submission_log(log_data)

    st.markdown(f"🧮 Readability: {readability} ({avg_words:.1f} w/s)")
    st.metric("Content",    f"{content_score}/10")
    st.metric("Grammar",    f"{grammar_score}/5")
    st.metric("Vocabulary", f"{vocab_score}/5")
    st.metric("Structure",  f"{structure_score}/5")
    st.markdown(f"**Total: {total}/25**")

    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: unique ratio {unique_ratio:.2f} ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    if gpt_results:
        st.markdown("**Grammar Suggestions:**")
        for i, line in enumerate(gpt_results, 1):
            st.markdown(f"{i}. {line}")

    hints = sorted(connectors_by_level.get(level, []))[:4]
    st.info(f"📝 Try connectors like: {', '.join(hints)}…")

    ann = annotate_text(student_text, gpt_results, adv, connectors_by_level, level)
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(ann, unsafe_allow_html=True)

    st.markdown("""
**What do the highlights mean?**

- <span style='background-color:#e15759; color:#fff'>Red</span>: Grammar error  
- <span style='background-color:#f1c232; color:#000'>Yellow</span>: Phrase is too long for your level  
- <span style='background-color:#6aa84f; color:#fff'>Green</span>: Connector word (like <i>und</i>, <i>aber</i>, <i>weil</i>)  
- <span style='background-color:#e69138; color:#fff'>Orange</span>: Passive voice or noun not capitalized  
- <span style='background-color:#cccccc; color:#000'>Gray</span>: Very long sentence (over 100 characters)  
- <span style='text-decoration:underline; color:#e15759'>Underlined</span>: Repeated word  
- <span style='border:1px solid #e15759'>Red border</span>: Double space or missing space after comma  
    """, unsafe_allow_html=True)

    # 🔍 What was highlighted and why
    st.markdown("### 🔍 What was highlighted and why")
    if gpt_results:
        st.markdown("- 🔴 Grammar errors: " + ", ".join(e.split("⇒")[0].strip(" `") for e in gpt_results))
    if adv:
        st.markdown("- 🟡 Too-long phrase(s): " + ", ".join(adv))
    if used_connectors:
        st.markdown("- 🟢 Connectors used: " + ", ".join(used_connectors))

    passives   = re.findall(r"\b(?:wird\s+\w+\s+von|ist\s+\w+\s+worden)\b", student_text, flags=re.I)
    long_sents = re.findall(r"([A-ZÄÖÜ][^\.!?]{100,}[\.!?])", student_text)
    noun_issues= re.findall(r"\b(?:der|die|das|ein|eine|mein|dein)\s+([a-zäöüß]+)\b", student_text, flags=re.I)
    ds         = re.findall(r" {2,}", student_text)
    mc         = re.findall(r",(?=[A-Za-zÖÜÄ])", student_text)
    repeats    = re.findall(r"\b(\w+)\s+\1\b", student_text, flags=re.I)

    if passives:    st.markdown("- 🟠 Passive voice flagged: " + ", ".join(passives))
    if long_sents:  st.markdown("- ⚪️ Long sentence(s): " + " | ".join(long_sents[:3]) + (" ..." if len(long_sents)>3 else ""))
    if noun_issues: st.markdown("- 🟠 Noun capitalization missing: " + ", ".join(noun_issues))
    if ds or mc:
        issues = []
        if ds: issues.append(f"{len(ds)} double space(s)")
        if mc: issues.append(f"{len(mc)} comma-space issue(s)")
        st.markdown("- 🔴 Punctuation issues: " + "; ".join(issues))
    if repeats:     st.markdown("- 🔴 Repeated words: " + ", ".join(sorted(set(repeats))))

    st.download_button("💾 Download feedback", data=feedback_text, file_name="feedback.txt")



