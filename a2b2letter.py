import re
import csv
import os
import streamlit as st
import pandas as pd
import openai
import stopwordsiso as stop
from datetime import datetime

TRAINING_DATA_PATH = "essay_training_data.csv"

st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")
st.title("📝 German Letter & Essay Checker – Learn Language Education Academy")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .css-18ni7ap.e8zbici2 {display: none;}
    .css-164nlkn.ezrtsby2 {display: none;}
    .css-1dp5vir.ezrtsby2 {display: none;}
    div[data-testid="stHeader"] {display: none;}
    </style>
    """, unsafe_allow_html=True)

VOCAB_PATH = "approved_vocab.csv"
CONNECTOR_PATH = "approved_connectors.csv"
LOG_PATH = "submission_log.csv"

# -- API Key --
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as environment variable.")
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# -- Stopwords and function words --
german_stopwords = stop.stopwords("de")
function_words = {w.lower() for w in {
    "ich","du","er","sie","wir","ihr","mir","mich",
    "der","die","das","ein","eine",
    "und","oder","aber","nicht","wie","wann","wo",
    "sein","haben","können","müssen","wollen","sollen",
    "bitte","viel","gut","sehr",
    "wer","was","wann","wo","warum","wie",
    "order"
}}
advanced_suggestions = {
    "rechnung":      "zahlung",
    "informationen": "angaben",
    "deutschkurs":   "kurs",
}

# --- Helpers for persistence ---
def load_vocab_from_csv():
    vocab = {"A1": set(), "A2": set()}
    if os.path.exists(VOCAB_PATH):
        with open(VOCAB_PATH, newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] in vocab:
                    vocab[row[0]].add(row[1].strip().lower())
    # Defaults for empty vocab files
    if not any(vocab.values()):
        vocab["A1"].update(["bitte", "danke", "tschüss", "möchten", "schreiben", "bezahlen"])
        vocab["A2"].update(["arbeiten", "lernen", "verstehen", "helfen"])
    return vocab

def save_vocab_to_csv(vocab):
    with open(VOCAB_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for level in ["A1", "A2"]:
            for word in sorted(vocab[level]):
                writer.writerow([level, word])

def load_connectors_from_csv():
    conns = {"A1": set(), "A2": set(), "B1": set(), "B2": set()}
    if os.path.exists(CONNECTOR_PATH):
        with open(CONNECTOR_PATH, newline='', encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) >= 2 and row[0] in conns:
                    items = [w.strip() for w in row[1].split(",") if w.strip()]
                    conns[row[0]].update(items)
    # Defaults
    if not any(conns.values()):
        conns = {
            "A1": {"weil","denn","ich möchte wissen","deshalb"},
            "A2": {"deshalb","deswegen","darum","trotzdem","obwohl","sobald","außerdem","zum Beispiel","und","aber","oder","erstens","zweitens","zum Schluss"},
            "B1": {"jedoch","allerdings","hingegen","trotzdem","dennoch","folglich","daher","demnach","infolgedenden","deshalb","damit","sofern","falls","währenddessen","inzwischen","mittlerweile","anschließend","schließlich","beispielsweise","zumal","wohingegen","erstens","zweitens","kurzum","zusammenfassend","einerseits","andererseits"},
            "B2": {"allerdings","dennoch","gleichwohl","demzufolge","mithin","ergo","sodass","obgleich","obschon","wenngleich","ungeachtet","indessen","nichtsdestotrotz","einerseits","andererseits","zumal","insofern","insoweit","demgemäß","zusammenfassend","abschließend","letztendlich"}
        }
    return conns

def save_connectors_to_csv(conns):
    with open(CONNECTOR_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for lvl in ["A1", "A2", "B1", "B2"]:
            writer.writerow([lvl, ", ".join(sorted(conns[lvl]))])

def load_student_codes():
    codes = set()
    if os.path.exists("student_codes.csv"):
        with open("student_codes.csv", newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            idx = headers.index('student_code') if headers and 'student_code' in headers else 0
            for row in reader:
                if len(row) > idx and row[idx].strip():
                    codes.add(row[idx].strip())
    return codes

def load_submission_log():
    data = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline='', encoding="utf-8") as f:
            for sid, count in csv.reader(f):
                data[sid] = int(count)
    return data
# --- Training Data Helpers ---
def save_for_training(student_id, level, task_type, task_num, student_text, gpt_results, feedback_text):
    """Append a student submission (with feedback) to the AI training CSV."""
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
    file_exists = os.path.exists(TRAINING_DATA_PATH)
    df = pd.DataFrame([row])
    if not file_exists or os.stat(TRAINING_DATA_PATH).st_size == 0:
        df.to_csv(TRAINING_DATA_PATH, index=False)
    else:
        df.to_csv(TRAINING_DATA_PATH, mode='a', header=False, index=False)

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

def render_collected_essays_for_training():
    """Display the AI training data download section in the dashboard."""
    st.subheader("Collected Essays for AI Training")
    download_training_data()

# --- Advanced vocab detection ---
def detect_advanced_vocab(text: str, level: str, approved_vocab) -> list[str]:
    text_norm = text.replace('\u2010','-').replace('\u2011','-')
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", text_norm)
    allowed = approved_vocab.get(level, set()) | german_stopwords | function_words
    flags = []
    for w in set(tokens):
        lw = w.lower().strip('-')
        if len(lw) <= 2 or lw in allowed or w.isupper():
            continue
        flags.append(w)
    return sorted(flags)

# --- GPT grammar check ---
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

# --- Annotation function ---
def annotate_text(student_text, gpt_results, adv, level):
    ann = student_text
    colors = {'Grammar': '#e15759', 'Advanced': '#f1c232'}

    # Highlight grammar errors
    if gpt_results:
        for line in gpt_results:
            if "⇒" in line:
                err = line.split("⇒")[0].strip(" `")
                if len(err) > 1:
                    pattern = r'(?i)\b' + re.escape(err) + r'\b'
                    def repl_grammar(m):
                        if "</span>" in m.group(0):
                            return m.group(0)
                        return f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>"
                    ann = re.sub(pattern, repl_grammar, ann)

    # Highlight advanced vocabulary (A1/A2)
    if level in ["A1", "A2"] and adv:
        for word in adv:
            pattern = rf'(?i)\b({re.escape(word)})\b'
            def repl_advanced(m):
                if "</span>" in m.group(0):
                    return m.group(0)
                return (f"<span title='Too advanced for {level}' "
                        f"style='background-color:{colors['Advanced']}; color:#000'>{m.group(1)}</span>")
            ann = re.sub(pattern, repl_advanced, ann)

    return ann.replace("\n", "  \n")
# --- A1 tasks (static) ---
a1_tasks = {
    1: {
        "task": "Schreiben Sie eine E-Mail an Ihren Arzt und sagen Sie Ihren Termin ab.",
        "points": [
            "Warum schreiben Sie?",
            "Sagen Sie: den Grund für die Absage.",
            "Fragen Sie: nach einem neuen Termin."
        ]
    },
    2: {
        "task": "Schreiben Sie eine Einladung an Ihren Freund zur Feier Ihres neuen Jobs.",
        "points": [
            "Warum schreiben Sie?",
            "Wann ist die Feier?",
            "Wer soll was mitbringen?"
        ]
    },
    3: {
        "task": "Schreiben Sie eine E-Mail an einen Freund und teilen Sie ihm mit, dass Sie ihn besuchen möchten.",
        "points": [
            "Warum schreiben Sie?",
            "Wann besuchen Sie ihn?",
            "Was möchten Sie zusammen machen?"
        ]
    },
    4: {
        "task": "Schreiben Sie eine E-Mail an Ihre Schule und fragen Sie nach einem Deutschkurs.",
        "points": [
            "Warum schreiben Sie?",
            "Was möchten Sie wissen?",
            "Wie kann die Schule antworten?"
        ]
    },
    5: {
        "task": "Schreiben Sie eine E-Mail an Ihre Vermieterin. Ihre Heizung ist kaputt.",
        "points": [
            "Warum schreiben Sie?",
            "Seit wann ist die Heizung kaputt?",
            "Was soll die Vermieterin tun?"
        ]
    },
    6: {
        "task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben eine neue Wohnung.",
        "points": [
            "Warum schreiben Sie?",
            "Wo ist die Wohnung?",
            "Was gefällt Ihnen?"
        ]
    },
    7: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben eine neue Arbeitsstelle.",
        "points": [
            "Warum schreiben Sie?",
            "Wo arbeiten Sie jetzt?",
            "Was machen Sie?"
        ]
    },
    8: {
        "task": "Schreiben Sie eine E-Mail an Ihren Lehrer. Sie können am Kurs nicht teilnehmen.",
        "points": [
            "Warum schreiben Sie?",
            "Warum kommen Sie nicht?",
            "Was möchten Sie?"
        ]
    },
    9: {
        "task": "Schreiben Sie eine E-Mail an die Bibliothek. Sie haben ein Buch verloren.",
        "points": [
            "Warum schreiben Sie?",
            "Welches Buch haben Sie verloren?",
            "Was möchten Sie wissen?"
        ]
    },
    10: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten mit ihr in den Urlaub fahren.",
        "points": [
            "Warum schreiben Sie?",
            "Wohin möchten Sie fahren?",
            "Was möchten Sie dort machen?"
        ]
    },
    11: {
        "task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Termin ändern.",
        "points": [
            "Warum schreiben Sie?",
            "Welcher Termin ist es?",
            "Wann haben Sie Zeit?"
        ]
    },
    12: {
        "task": "Schreiben Sie eine E-Mail an Ihren Bruder. Sie machen eine Party.",
        "points": [
            "Warum schreiben Sie?",
            "Wann ist die Party?",
            "Was soll Ihr Bruder mitbringen?"
        ]
    },
    13: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie sind krank.",
        "points": [
            "Warum schreiben Sie?",
            "Was machen Sie heute nicht?",
            "Was sollen Sie tun?"
        ]
    },
    14: {
        "task": "Schreiben Sie eine E-Mail an Ihre Nachbarn. Sie machen Urlaub.",
        "points": [
            "Warum schreiben Sie?",
            "Wie lange sind Sie weg?",
            "Was sollen die Nachbarn tun?"
        ]
    },
    15: {
        "task": "Schreiben Sie eine E-Mail an Ihre Deutschlehrerin. Sie möchten eine Prüfung machen.",
        "points": [
            "Warum schreiben Sie?",
            "Welche Prüfung möchten Sie machen?",
            "Wann möchten Sie die Prüfung machen?"
        ]
    },
    16: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben einen neuen Computer gekauft.",
        "points": [
            "Warum schreiben Sie?",
            "Wo haben Sie den Computer gekauft?",
            "Was gefällt Ihnen besonders?"
        ]
    },
    17: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen Sport machen.",
        "points": [
            "Warum schreiben Sie?",
            "Welchen Sport möchten Sie machen?",
            "Wann können Sie?"
        ]
    },
    18: {
        "task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie brauchen Hilfe beim Umzug.",
        "points": [
            "Warum schreiben Sie?",
            "Wann ist der Umzug?",
            "Was soll Ihr Freund machen?"
        ]
    },
    19: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten ein Fest organisieren.",
        "points": [
            "Warum schreiben Sie?",
            "Wo soll das Fest sein?",
            "Was möchten Sie machen?"
        ]
    },
    20: {
        "task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen kochen.",
        "points": [
            "Warum schreiben Sie?",
            "Was möchten Sie kochen?",
            "Wann möchten Sie kochen?"
        ]
    },
    21: {
        "task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben einen neuen Job.",
        "points": [
            "Warum schreiben Sie?",
            "Wo arbeiten Sie jetzt?",
            "Was machen Sie?"
        ]
    },
    22: {
        "task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Deutschkurs besuchen.",
        "points": [
            "Warum schreiben Sie?",
            "Wann möchten Sie den Kurs besuchen?",
            "Was möchten Sie noch wissen?"
        ]
    }
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

def render_collected_essays_for_training():
    """Display the AI training data download section in the dashboard."""
    st.subheader("Collected Essays for AI Training")
    download_training_data()

# --- Teacher Dashboard ---
if teacher_mode and page == "Teacher Dashboard":
    st.header("📊 Teacher Dashboard")

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
            items = {c.strip() for c in new_conns.split(",") if c.strip()}
            connectors[lvl] = items
            save_connectors_to_csv(connectors)
            st.success(f"✅ {lvl} connectors updated.")

    # --- Student Codes Editor ---
    st.subheader("Student Codes")
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
        
    # Submission Log
    st.subheader("Submission Log")
    df_log = pd.DataFrame(load_submission_log().items(), columns=["Student Code","Submissions"])
    st.dataframe(df_log)
    st.download_button("💾 Download Log", data=df_log.to_csv(index=False).encode('utf-8'),
                       file_name="submission_log.csv", mime='text/csv')

    render_collected_essays_for_training()
    st.stop()

# --- STUDENT VIEW ---
# Always load latest data
approved_vocab = load_vocab_from_csv()
student_codes = load_student_codes()
log_data = load_submission_log()
connectors_by_level = load_connectors_from_csv()

level = st.selectbox("Select your level", ["A1","A2","B1","B2"])
tasks = ["Formal Letter","Informal Letter"]
if level in ("B1","B2"): tasks.append("Opinion Essay")
task_type = st.selectbox("Select task type", tasks)

st.markdown("### ✍️ Structure & Tips")
with st.expander("✍️ Writing Tips and Usage Advice"):
    if level == "A1":
        st.markdown(
            "- Use simple present tense (ich bin, ich habe, ich wohne...)\n"
            "- Keep sentences short and clear\n"
            "- Use basic connectors und, aber, weil\n"
            "- Avoid complex verbs or modal structures\n"
            "- Always start sentences with a capital letter"
        )
    elif level == "A2":
        st.markdown("- Explain reasons using weil and denn\n- Add time expressions (z.B. am Montag, um 8 Uhr)\n- Include polite forms like ich möchte, könnten Sie?")
    elif level == "B1":
        st.markdown("- Include both pros and cons in essays\n- Use connectors like einerseits...andererseits, deshalb, trotzdem\n- Vary sentence structure with subordinates")
    else:
        st.markdown("- Support opinions with examples and evidence\n- Use passive voice and indirect speech when appropriate\n- Include complex structures with relative and conditional clauses")

student_id = st.text_input("Enter your student code:")
if not student_id:
    st.warning("Please enter your student code.")
    st.stop()
if student_id not in student_codes:
    st.error("❌ You are not authorized to use this app.")
    st.stop()

subs = log_data.get(student_id, 0)
max_subs = 40 if level == 'A1' else 45
if subs >= max_subs:
    st.warning(f"⚠️ You have reached the maximum of {max_subs} submissions.")
    st.stop()
if subs >= max_subs - 12:
    st.info("⏳ You have used most of your submission chances. Review carefully!")

task_num = None
task = None
if level == "A1":
    task_num = st.number_input(f"Choose a Schreiben task number (1 to {len(a1_tasks)})", 1, len(a1_tasks), 1)
    try:
        task = a1_tasks[int(task_num)]
        st.markdown(f"### Aufgabe {task_num}: {task['task']}")
        st.markdown("**Points:**")
        for p in task['points']:
            st.markdown(f"- {p}")
    except KeyError:
        st.error("Invalid task number.")

with st.form("feedback_form"):
    student_text = st.text_area("✏️ Write your letter or essay below:", height=300)
    submit = st.form_submit_button("✅ Submit for Feedback")

def score_text(student_text, level, gpt_results, adv):
    words = re.findall(r"\w+", student_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    sentences = re.split(r'[.!?]', student_text)
    avg_words = len(words) / max(1, len([s for s in sentences if s.strip()]))
    readability = "Easy" if avg_words <= 12 else "Medium" if avg_words <= 17 else "Hard"
    content_score = 10
    grammar_score = max(1, 5 - len(gpt_results))
    vocab_score = min(5, int((len(set(words))/len(words))*5))
    if adv:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score
    return content_score, grammar_score, vocab_score, structure_score, total, unique_ratio, avg_words, readability

def generate_feedback_text(level, task_type, task, content_score, grammar_score, vocab_score, structure_score, total, gpt_results, adv, used, student_text):
    feedback_text = f"""Your Feedback – {task_type} ({level})
Task: {task['task'] if task else ''}
Scores:
- Content: {content_score}/10
- Grammar: {grammar_score}/5
- Vocabulary: {vocab_score}/5
- Structure: {structure_score}/5
Total: {total}/25

Grammar Suggestions:
{chr(10).join(gpt_results) if gpt_results else 'No major grammar errors detected.'}

Advanced Vocabulary:
{', '.join(adv) if adv else 'None'}

Connectors Used:
{', '.join(used) if used else 'None'}

Your Text:
{student_text}
"""
    return feedback_text

if submit:
    with st.spinner("🔄 Processing your submission, please wait…"):
        if not student_text.strip():
            st.warning("Please enter your text before submitting.")
            st.stop()

        # 1. GPT grammar check
        gpt_results = grammar_check_with_gpt(student_text)
        adv = detect_advanced_vocab(student_text, level, approved_vocab) if level in ("A1","A2") else []

        # 2. Scoring
        content_score, grammar_score, vocab_score, structure_score, total, unique_ratio, avg_words, readability = \
            score_text(student_text, level, gpt_results, adv)

        # 3. Generate the full feedback text
        feedback_text = generate_feedback_text(
            level, task_type, task,
            content_score, grammar_score, vocab_score, structure_score, total,
            gpt_results, adv, [], student_text
        )

        # 4. **Save this submission for AI training**
        save_for_training(
            student_id=student_id,
            level=level,
            task_type=task_type,
            task_num=task_num,
            student_text=student_text,
            gpt_results=gpt_results,
            feedback_text=feedback_text
        )

    # 5. Update submission log
    log_data[student_id] = log_data.get(student_id, 0) + 1
    with open(LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for k, v in log_data.items():
            writer.writerow([k, v])

    # 6. Warnings and metrics display
    if adv:
        st.warning(f"⚠️ The following words may be too advanced: {', '.join(adv)}")

    st.markdown(f"🧮 Readability: {readability} ({avg_words:.1f} w/s)")
    st.metric("Content", f"{content_score}/10")
    st.metric("Grammar", f"{grammar_score}/5")
    st.metric("Vocabulary", f"{vocab_score}/5")
    st.metric("Structure", f"{structure_score}/5")
    st.markdown(f"**Total: {total}/25**")

    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: ratio {unique_ratio:.2f}, penalties ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    if gpt_results:
        st.markdown("**Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

    if not teacher_mode:
        hints = sorted(connectors_by_level.get(level, []))[:4]
        st.info(f"📝 Try connectors like: {', '.join(hints)}…")

    conns = connectors_by_level.get(level, set())
    used = [c for c in conns if c in student_text.lower()]
    if used:
        st.success(f"✅ You used connectors: {', '.join(used)}")
    else:
        st.info(f"📝 Consider using more connectors for clarity.")

    # --- Highlight grammar errors and advanced words ---
    ann = student_text
    colors = {'Grammar': '#e15759', 'Advanced': '#f1c232'}

    # Highlight grammar errors
    if gpt_results:
        for line in gpt_results:
            if "⇒" in line:
                err = line.split("⇒")[0].strip(" `")
                pattern = re.escape(err)
                ann = re.sub(
                    pattern,
                    f"<span style='background-color:{colors['Grammar']}; color:#fff'>{err}</span>",
                    ann,
                    flags=re.I
                )

    # Highlight advanced vocabulary only for A1 and A2
    if level in ["A1", "A2"] and adv:
        for word in adv:
            pattern = rf"\b({re.escape(word)})\b(?![^<]*</span>)"
            ann = re.sub(
                pattern,
                rf"<span title='Too advanced for {level}' style='background-color:{colors['Advanced']}; color:#000'>\1</span>",
                ann,
                flags=re.I
            )
        st.markdown("**📚 Advanced Vocabulary Used:**")
        for word in adv:
            st.markdown(f"- {word} _(not recommended for {level})_ ")

    safe_ann = ann.replace("\n", "  \n")

    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(safe_ann, unsafe_allow_html=True)


    feedback_text = generate_feedback_text(level, task_type, task, content_score, grammar_score, vocab_score, structure_score, total, gpt_results, adv, used, student_text)
    st.download_button("💾 Download feedback", data=feedback_text, file_name="feedback.txt")
