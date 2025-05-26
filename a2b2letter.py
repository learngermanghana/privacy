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

VOCAB_PATH      = "approved_vocab.csv"
CONNECTOR_PATH  = "approved_connectors.csv"
LOG_PATH        = "submission_log.csv"
STUDENT_CODES_PATH = "student_codes.csv"

# -- API Key --
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as environment variable.")
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# -- Stopwords and function words --
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
            for lvl, word in csv.reader(f):
                vocab[lvl].add(word.strip().lower())
    if not any(vocab.values()):
        vocab["A1"].update(["bitte","danke","tschüss","möchten","schreiben","bezahlen"])
        vocab["A2"].update(["arbeiten","lernen","verstehen","helfen"])
    return vocab

def load_connectors_from_csv():
    conns = {l:set() for l in ["A1","A2","B1","B2"]}
    if os.path.exists(CONNECTOR_PATH):
        with open(CONNECTOR_PATH, newline='', encoding="utf-8") as f:
            for lvl, items in csv.reader(f):
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

def load_submission_log():
    data = {}
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, newline='', encoding="utf-8") as f:
            for sid, count in csv.reader(f):
                data[sid] = int(count)
    return data

# --- Training Data Helpers ---
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
    file_exists = os.path.exists(TRAINING_DATA_PATH)
    df = pd.DataFrame([row])
    if not file_exists or os.stat(TRAINING_DATA_PATH).st_size == 0:
        df.to_csv(TRAINING_DATA_PATH, index=False)
    else:
        df.to_csv(TRAINING_DATA_PATH, mode='a', header=False, index=False)

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

# --- Advanced vocab detection ---
def detect_advanced_vocab(text: str, level: str, approved_vocab, connectors) -> list[str]:
    tokens = re.findall(r"[A-Za-zÄÖÜäöüß\-]+", text)
    allowed = (
        approved_vocab.get(level, set())
        | german_stopwords
        | function_words
        | set(connectors)
    )
    adv = set()
    for w in tokens:
        lw = w.lower().strip('-')
        if lw in allowed:      continue
        if w != w.lower():     continue
        if len(lw) <= 6:       continue
        adv.add(lw)
    return sorted(adv)

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

# --- Annotation function (updated) ---
def annotate_text(student_text, gpt_results, adv, level):
    ann = student_text
    colors = {
        'Grammar':   '#e15759',
        'Advanced':  '#f1c232',
        'Connector': '#6aa84f',
        'Passive':   '#f1c232',
        'LongSent':  '#cccccc',
        'Noun':      '#e69138'
    }

    # 1) Grammar errors
    for line in gpt_results or []:
        if "⇒" in line:
            err = line.split("⇒")[0].strip(" `")
            if len(err) > 1:
                pat = rf"(?i)\b{re.escape(err)}\b"
                ann = re.sub(pat,
                             lambda m: f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>",
                             ann)

    # 2) Advanced vocabulary
    if level in ("A1","A2") and adv:
        for w in adv:
            pat = rf"(?i)\b{re.escape(w)}\b"
            ann = re.sub(pat,
                         lambda m: f"<span title='Too advanced for {level}' style='background-color:{colors['Advanced']}; color:#000'>{m.group(0)}</span>",
                         ann)

    # 3) Approved connectors
    for conn in load_connectors_from_csv().get(level, []):
        pat = rf"(?i)\b{re.escape(conn)}\b"
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Connector']}; color:#fff'>{m.group(0)}</span>",
                     ann)

    # 4) Passive voice
    for pat in [r"\bwird\s+\w+\s+von\b", r"\bist\s+\w+\s+worden\b"]:
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Passive']}; color:#000'>{m.group(0)}</span>",
                     ann, flags=re.I)

    # 5) Long sentences (>100 chars)
    ann = re.sub(r"([A-ZÄÖÜ][^\.!?]{100,}[\.!?])",
                 lambda m: f"<span style='background-color:{colors['LongSent']}; color:#000'>{m.group(0)}</span>",
                 ann)

    # 6) Missing noun capitalization
    for det in [' der',' die',' das',' ein',' eine',' mein',' dein']:
        pat = rf"(?<={det}\s)([a-zäöüß]+)\b"
        ann = re.sub(pat,
                     lambda m: f"<span style='background-color:{colors['Noun']}; color:#fff'>{m.group(0)}</span>",
                     ann)

    # 7a) Double spaces
    ann = re.sub(r" {2,}",
                 lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
                 ann)
    # 7b) Missing space after comma
    ann = re.sub(r",(?=[A-Za-zÖÜÄ])",
                 lambda m: f"<span style='border:1px solid {colors['Grammar']}'>{m.group(0)}</span>",
                 ann)

    # 8) Repeated words
    ann = re.sub(r"\b(\w+)\s+\1\b",
                 lambda m: f"<span style='text-decoration:underline; color:{colors['Grammar']}'>{m.group(0)}</span>",
                 ann, flags=re.I)

    return ann.replace("\n", "  \n")

# --- A1 tasks (static) ---
a1_tasks = {
    1: {"task":"Schreiben Sie eine E-Mail an Ihren Arzt und sagen Sie Ihren Termin ab.",
        "points":["Warum schreiben Sie?","Sagen Sie: den Grund für die Absage.","Fragen Sie: nach einem neuen Termin."]},
    2: {"task":"Schreiben Sie eine Einladung an Ihren Freund zur Feier Ihres neuen Jobs.",
        "points":["Warum schreiben Sie?","Wann ist die Feier?","Wer soll was mitbringen?"]},
    3: {"task":"Schreiben Sie eine E-Mail an einen Freund und teilen Sie ihm mit, dass Sie ihn besuchen möchten.",
        "points":["Warum schreiben Sie?","Wann besuchen Sie ihn?","Was möchten Sie zusammen machen?"]},
    4: {"task":"Schreiben Sie eine E-Mail an Ihre Schule und fragen Sie nach einem Deutschkurs.",
        "points":["Warum schreiben Sie?","Was möchten Sie wissen?","Wie kann die Schule antworten?"]},
    5: {"task":"Schreiben Sie eine E-Mail an Ihre Vermieterin. Ihre Heizung ist kaputt.",
        "points":["Warum schreiben Sie?","Seit wann ist die Heizung kaputt?","Was soll die Vermieterin tun?"]},
    6: {"task":"Schreiben Sie eine E-Mail an Ihren Freund. Sie haben eine neue Wohnung.",
        "points":["Warum schreiben Sie?","Wo ist die Wohnung?","Was gefällt Ihnen?"]},
    7: {"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben eine neue Arbeitsstelle.",
        "points":["Warum schreiben Sie?","Wo arbeiten Sie jetzt?","Was machen Sie?"]},
    8: {"task":"Schreiben Sie eine E-Mail an Ihren Lehrer. Sie können am Kurs nicht teilnehmen.",
        "points":["Warum schreiben Sie?","Warum kommen Sie nicht?","Was möchten Sie?"]},
    9: {"task":"Schreiben Sie eine E-Mail an die Bibliothek. Sie haben ein Buch verloren.",
        "points":["Warum schreiben Sie?","Welches Buch haben Sie verloren?","Was möchten Sie wissen?"]},
    10:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten mit ihr in den Urlaub fahren.",
        "points":["Warum schreiben Sie?","Wohin möchten Sie fahren?","Was möchten Sie dort machen?"]},
    11:{"task":"Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Termin ändern.",
        "points":["Warum schreiben Sie?","Welcher Termin ist es?","Wann haben Sie Zeit?"]},
    12:{"task":"Schreiben Sie eine E-Mail an Ihren Bruder. Sie machen eine Party.",
        "points":["Warum schreiben Sie?","Wann ist die Party?","Was soll Ihr Bruder mitbringen?"]},
    13:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie sind krank.",
        "points":["Warum schreiben Sie?","Was machen Sie heute nicht?","Was sollen Sie tun?"]},
    14:{"task":"Schreiben Sie eine E-Mail an Ihre Nachbarn. Sie machen Urlaub.",
        "points":["Warum schreiben Sie?","Wie lange sind Sie weg?","Was sollen die Nachbarn tun?"]},
    15:{"task":"Schreiben Sie eine E-Mail an Ihre Deutschlehrerin. Sie möchten eine Prüfung machen.",
        "points":["Warum schreiben Sie?","Welche Prüfung möchten Sie machen?","Wann möchten Sie die Prüfung machen?"]},
    16:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben einen neuen Computer gekauft.",
        "points":["Warum schreiben Sie?","Wo haben Sie den Computer gekauft?","Was gefällt Ihnen besonders?"]},
    17:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen Sport machen.",
        "points":["Warum schreiben Sie?","Welchen Sport möchten Sie machen?","Wann können Sie?"]},
    18:{"task":"Schreiben Sie eine E-Mail an Ihren Freund. Sie brauchen Hilfe beim Umzug.",
        "points":["Warum schreiben Sie?","Wann ist der Umzug?","Was soll Ihr Freund machen?"]},
    19:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten ein Fest organisieren.",
        "points":["Warum schreiben Sie?","Wo soll das Fest sein?","Was möchten Sie machen?"]},
    20:{"task":"Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen kochen.",
        "points":["Warum schreiben Sie?","Was möchten Sie kochen?","Wann möchten Sie kochen?"]},
    21:{"task":"Schreiben Sie eine E-Mail an Ihren Freund. Sie haben einen neuen Job.",
        "points":["Warum schreiben Sie?","Wo arbeiten Sie jetzt?","Was machen Sie?"]},
    22:{"task":"Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Deutschkurs besuchen.",
        "points":["Warum schreiben Sie?","Wann möchten Sie den Kurs besuchen?","Was möchten Sie noch wissen?"]},
}

# --- Teacher Settings ---
st.sidebar.header("🔧 Teacher Settings")
teacher_password = st.sidebar.text_input("🔒 Enter teacher password", type="password")
teacher_mode = (teacher_password == "Felix029")
if teacher_mode:
    page = st.sidebar.radio("Go to:", ["Student View", "Teacher Dashboard"])
else:
    page = "Student View"

# Reusable training-data download
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
    st.subheader("📊 Collected Essays for AI Training")
    download_training_data()

# --- Teacher Dashboard ---
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
    st.subheader("Edit Approved Connectors (A1–B2)")
    connectors = load_connectors_from_csv()
    for lvl in ["A1", "A2", "B1", "B2"]:
        current = ", ".join(sorted(connectors.get(lvl, set())))
        new_conns = st.text_area(f"{lvl} Connectors (comma-separated):", current, key=f"conn_{lvl}")
        if st.button(f"Update {lvl} Connectors", key=f"btn_conn_{lvl}"):
            items = {c.strip() for c in new_conns.split(",") if c.strip()}
            connectors[lvl] = items
            save_connectors_to_csv(connectors)
            st.success(f"✅ {lvl} connectors updated.")

    # ---- Student Codes Editor ----
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
        
    # ---- Submission Log ----
    st.subheader("Submission Log")
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

    # ---- Training Data Download ----
    render_collected_essays_for_training()

    st.stop()

# --- STUDENT VIEW ---
# Always load latest data
approved_vocab      = load_vocab_from_csv()
student_codes       = load_student_codes()
log_data            = load_submission_log()
connectors_by_level = load_connectors_from_csv()

# Level & Task Type selection
level = st.selectbox("Select your level", ["A1","A2","B1","B2"])
tasks = ["Formal Letter","Informal Letter"]
if level in ("B1","B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select task type", tasks)

# Writing Tips
st.markdown("### ✍️ Structure & Tips")
with st.expander("✍️ Writing Tips and Usage Advice"):
    if level == "A1":
        st.markdown(
            "- Use simple present tense (ich bin, ich habe, ich wohne...)  \n"
            "- Keep sentences short and clear  \n"
            "- Use basic connectors und, aber, weil  \n"
            "- Avoid complex verbs or modal structures  \n"
            "- Always start sentences with a capital letter"
        )
    elif level == "A2":
        st.markdown(
            "- Explain reasons using weil and denn  \n"
            "- Add time expressions (z.B. am Montag, um 8 Uhr)  \n"
            "- Include polite forms like ich möchte, könnten Sie?"
        )
    elif level == "B1":
        st.markdown(
            "- Include both pros and cons in essays  \n"
            "- Use connectors like einerseits…andererseits, deshalb, trotzdem  \n"
            "- Vary sentence structure with subordinate clauses"
        )
    else:
        st.markdown(
            "- Support opinions with examples and evidence  \n"
            "- Use passive voice and indirect speech when appropriate  \n"
            "- Include relative and conditional clauses"
        )

# Student authentication & submission limit
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

# A1-specific task selection
task_num = None
task     = None
if level == "A1":
    task_num = st.number_input(f"Choose a Schreiben task number (1–{len(a1_tasks)})",
                               1, len(a1_tasks), 1)
    task = a1_tasks.get(task_num)
    st.markdown(f"### Aufgabe {task_num}: {task['task']}")
    st.markdown("**Points:**")
    for p in task["points"]:
        st.markdown(f"- {p}")

# Submission form
with st.form("feedback_form"):
    student_text = st.text_area("✏️ Write your letter or essay below:", height=300)
    submit       = st.form_submit_button("✅ Submit for Feedback")

# On form submit
if submit:
    with st.spinner("🔄 Processing your submission…"):
        if not student_text.strip():
            st.warning("Please enter your text before submitting.")
            st.stop()

        # 1) GPT grammar check
        gpt_results = grammar_check_with_gpt(student_text)

        # 2) Advanced vocab detection
        adv = detect_advanced_vocab(
            student_text,
            level,
            approved_vocab,
            connectors_by_level.get(level, [])
        ) if level in ("A1","A2") else []

        # 3) Compute scores
        (content_score, grammar_score, vocab_score,
         structure_score, total,
         unique_ratio, avg_words,
         readability) = score_text(student_text, level, gpt_results, adv)

        # 4) Prepare feedback text
        feedback_text = generate_feedback_text(
            level, task_type, task, content_score, grammar_score,
            vocab_score, structure_score, total,
            gpt_results, adv, [], student_text
        )

        # 5) Save for AI training
        save_for_training(
            student_id=student_id,
            level=level,
            task_type=task_type,
            task_num=task_num,
            student_text=student_text,
            gpt_results=gpt_results,
            feedback_text=feedback_text
        )

        # 6) Update and persist submission log
        log_data[student_id] = subs + 1
        save_submission_log(log_data)

    # Display metrics
    if adv:
        st.warning(f"⚠️ Too-advanced words: {', '.join(adv)}")

    st.markdown(f"🧮 Readability: {readability} ({avg_words:.1f} w/s)")
    st.metric("Content",    f"{content_score}/10")
    st.metric("Grammar",    f"{grammar_score}/5")
    st.metric("Vocabulary", f"{vocab_score}/5")
    st.metric("Structure",  f"{structure_score}/5")
    st.markdown(f"**Total: {total}/25**")

    # Explain scores
    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: unique ratio {unique_ratio:.2f} ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    # Grammar suggestions list
    if gpt_results:
        st.markdown("**Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

    # Connector hints
    hints = sorted(connectors_by_level.get(level, []))[:4]
    st.info(f"📝 Try connectors like: {', '.join(hints)}…")

    # Annotated text
    ann = annotate_text(student_text, gpt_results, adv, level)
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(ann, unsafe_allow_html=True)

    # 🔍 What was highlighted and why
    st.markdown("### 🔍 What was highlighted and why")

    # 1. Grammar errors (red)
    if gpt_results:
        st.markdown(
            "- 🔴 **Grammar errors** (" + str(len(gpt_results)) + "): " +
            ", ".join(line.split("⇒")[0].strip(" `") for line in gpt_results)
        )

    # 2. Advanced vocab (yellow)
    if adv:
        st.markdown("- 🟡 **Too-advanced words**: " + ", ".join(adv))

    # 3. Approved connectors (green)
    used_connectors = [
        c for c in connectors_by_level.get(level, [])
        if c.lower() in student_text.lower()
    ]
    if used_connectors:
        st.markdown("- 🟢 **Connectors used correctly**: " + ", ".join(used_connectors))

    # 4. Passive-voice (orange)
    passives = re.findall(
        r"\b(?:wird\s+\w+\s+von|ist\s+\w+\s+worden)\b",
        student_text, flags=re.I
    )
    if passives:
        st.markdown("- 🟠 **Passive constructions flagged**: " + ", ".join(passives))

    # 5. Long sentences (gray)
    long_sents = re.findall(
        r"([A-ZÄÖÜ][^\.!?]{100,}[\.!?])",
        student_text
    )
    if long_sents:
        st.markdown(
            "- ⚪️ **Long sentence(s)**: " +
            " | ".join(long_sents[:3]) +
            (" ..." if len(long_sents) > 3 else "")
        )

    # 6. Noun capitalization issues (orange)
    noun_issues = re.findall(
        r"(?<=(?: der| die| das| ein| eine| mein| dein)\s)([a-zäöüß]+)\b",
        student_text
    )
    if noun_issues:
        st.markdown("- 🟠 **Noun capitalization missing**: " + ", ".join(noun_issues))

    # 7. Punctuation issues (red)
    double_spaces = re.findall(r" {2,}", student_text)
    missing_comma_space = re.findall(r",(?=[A-Za-zÖÜÄ])", student_text)
    if double_spaces or missing_comma_space:
        issues = []
        if double_spaces:
            issues.append(f"{len(double_spaces)} double-space(s)")
        if missing_comma_space:
            issues.append(f"{len(missing_comma_space)} comma-space issue(s)")
        st.markdown("- 🔴 **Punctuation issues**: " + "; ".join(issues))

    # 8. Repeated words (underline)
    repeats = re.findall(r"\b(\w+)\s+\1\b", student_text, flags=re.I)
    if repeats:
        st.markdown("- 🔴 **Repeated words**: " + ", ".join(sorted(set(repeats))))

    # Download feedback
    st.download_button(
        "💾 Download feedback",
        data=feedback_text,
        file_name="feedback.txt"
    )
