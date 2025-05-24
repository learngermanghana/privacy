import re
import csv
import os
import streamlit as st
import pandas as pd
import openai
import stopwordsiso as stop
from datetime import datetime

st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")
st.title("📝 German Letter & Essay Checker – Learn Language Education Academy")

# --- File paths ---
CONNECTOR_PATH = "approved_connectors.csv"
LOG_PATH = "submission_log.csv"
TRAINING_DATA_PATH = "essay_training_data.csv"
STUDENT_CODES_PATH = "student_codes.csv"

# --- API Key ---
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as environment variable.")
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# --- Stopwords and function words ---
german_stopwords = stop.stopwords("de")
function_words = {
    w.lower() for w in {
        "ich","du","er","sie","wir","ihr","mir","mich",
        "der","die","das","ein","eine",
        "und","oder","aber","nicht","wie","wann","wo",
        "sein","haben","können","müssen","wollen","sollen",
        "bitte","viel","gut","sehr",
        "wer","was","wann","wo","warum","wie",
        "order"
    }
}

# --- Approved vocabulary by level (hard-coded) ---
approved_vocab = {
    "A1": {
        "Anfrage","Anmelden","Terminen","Preisen","Kreditkarte","absagen",
        "anfangen","vereinbaren","übernachten","Rechnung","Informationen",
        "Anruf","antworten","Gebühr","buchen","eintragen","mitnehmen",
        "Unterschrift","Untersuchung","Unfall","abholen","abgeben",
        "mitteilen","erreichen","eröffnen","reservieren","verschieben",
        "freundlichen","besuchen","Abendessen","Restaurant",
        "bitte","danke","Entschuldigung","Hallo","Tschüss",
        "Name","Adresse","Telefonnummer","Straße","Postleitzahl",
        "Bahn","Bus","Auto","Fahrrad",
        "Apotheke","Supermarkt","Bäckerei",
        "heute","morgen","jetzt","später",
        "schreiben","lesen","sehen","hören"
    },
    "A2": {
        "verstehen","arbeiten","lernen","besuchen","fahren","lesen",
        "helfen","sprechen","finden","tragen","essen","geben",
        "wohnen","spielen","anmelden","krankenhaus","trainingszeiten",
        "kosten","Termin","Ausweis","Führerschein","Öffnungszeiten",
        "verabreden","verschieben","absagen","einladen","Reparatur",
        "Schlüssel","Nachricht","E-Mail","Reise","Urlaub","Hotel",
        "Bahnhof","Flughafen","schmecken","bestellen","bezahlen",
        "trinken","kochen","Kollege","Chef","Arbeit","Stelle","Firma"
    }
}

# --- GPT‐based grammar check ---
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

# --- Detect advanced vocabulary (A1/A2 only) ---
def detect_advanced_vocab(text: str, level: str) -> list[str]:
    """
    Flags any word not in approved_vocab[level].
    """
    tokens = re.findall(r"\w+", text)
    allowed = approved_vocab.get(level, set()) | german_stopwords | function_words
    return sorted(w for w in set(tokens) if w not in allowed)

# --- Helpers for connectors, students, logs, training data ---
def load_connectors_from_csv():
    conns = {"A1": set(), "A2": set(), "B1": set(), "B2": set()}
    if os.path.exists(CONNECTOR_PATH):
        with open(CONNECTOR_PATH, newline='', encoding="utf-8") as f:
            for lvl, conns_str in csv.reader(f):
                if lvl in conns:
                    conns[lvl] = {c.strip() for c in conns_str.split(",") if c.strip()}
    if not any(conns.values()):
        conns = {
            "A1": {"weil","denn","ich möchte wissen","deshalb"},
            "A2": {"deshalb","deswegen","darum","trotzdem","obwohl","sobald","außerdem","zum Beispiel","und","aber","oder","erstens","zweitens","zum Schluss"},
            "B1": {"jedoch","allerdings","hingegen","trotzdem","dennoch","folglich","daher","demnach","deshalb","damit","sofern","falls","währenddessen","inzwischen","mittlerweile","anschließend","schließlich","beispielsweise","zumal","wohingegen","erstens","zweitens","kurzum","zusammenfassend","einerseits","andererseits"},
            "B2": {"allerdings","dennoch","gleichwohl","demzufolge","mithin","ergo","sodass","obgleich","obschon","wenngleich","ungeachtet","indessen","nichtsdestotrotz","einerseits","andererseits","zumal","insofern","insoweit","demgemäß","zusammenfassend","abschließend","letztendlich"}
        }
    return conns

def save_connectors_to_csv(conns):
    with open(CONNECTOR_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for lvl in ["A1","A2","B1","B2"]:
            writer.writerow([lvl, ", ".join(sorted(conns[lvl]))])

def load_student_codes():
    codes = set()
    if os.path.exists(STUDENT_CODES_PATH):
        with open(STUDENT_CODES_PATH, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)
            idx = headers.index("student_code") if headers and "student_code" in headers else 0
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

def save_for_training(student_id, level, task_type, task_num, student_text, gpt_results, feedback_text):
    row = {
        "timestamp": datetime.now(),
        "student_id": student_id,
        "level": level,
        "task_type": task_type,
        "task_num": task_num,
        "original_text": student_text,
        "gpt_grammar_feedback": "\n".join(gpt_results),
        "full_feedback": feedback_text
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRAINING_DATA_PATH):
        df.to_csv(TRAINING_DATA_PATH, index=False)
    else:
        df.to_csv(TRAINING_DATA_PATH, mode="a", header=False, index=False)

def download_training_data():
    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, "rb") as f:
            st.download_button(
                "⬇️ Download All Submissions",
                data=f,
                file_name="essay_training_data.csv",
                mime="text/csv"
            )
    else:
        st.info("No training data collected yet.")

# --- Annotation helper ---
def annotate_text(student_text, gpt_results, adv, level):
    ann = student_text
    colors = {"Grammar":"#e15759","Advanced":"#f1c232"}
    # Highlight grammar
    for line in gpt_results:
        if "⇒" in line:
            err = line.split("⇒")[0].strip(" `")
            pattern = r"(?i)\b" + re.escape(err) + r"\b"
            ann = re.sub(pattern,
                         lambda m: f"<span style='background-color:{colors['Grammar']}; color:#fff'>{m.group(0)}</span>",
                         ann)
    # Highlight advanced vocab
    if level in ("A1","A2"):
        for w in adv:
            pattern = rf"(?i)\b({re.escape(w)})\b"
            ann = re.sub(pattern,
                         lambda m: f"<span title='Too advanced for {level}' style='background-color:{colors['Advanced']}; color:#000'>{m.group(1)}</span>",
                         ann)
    return ann.replace("\n","  \n")


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
page = st.sidebar.radio("Go to:", ["Student View","Teacher Dashboard"] if teacher_mode else ["Student View"])

# --- Teacher Dashboard ---
if teacher_mode and page == "Teacher Dashboard":
    st.header("📊 Teacher Dashboard")

    # Approved Vocabulary Editor
    st.subheader("Edit Approved Vocabulary (A1/A2)")
    vocab = load_vocab_from_csv()
    for lvl in ["A1","A2"]:
        txt = ", ".join(sorted(vocab[lvl]))
        new = st.text_area(f"{lvl} Vocabulary:", txt, key=f"vocab_{lvl}")
        if st.button(f"Save {lvl}", key=f"save_vocab_{lvl}"):
            vocab[lvl] = {w.strip().lower() for w in new.split(",") if w.strip()}
            save_vocab_to_csv(vocab)
            st.success(f"{lvl} vocabulary updated.")

    # Approved Connectors Editor
    st.subheader("Edit Approved Connectors (A1–B2)")
    connectors = load_connectors_from_csv()
    for lvl in ["A1","A2","B1","B2"]:
        txt = ", ".join(sorted(connectors[lvl]))
        new = st.text_area(f"{lvl} Connectors:", txt, key=f"conn_{lvl}")
        if st.button(f"Save {lvl} Connectors", key=f"save_conn_{lvl}"):
            connectors[lvl] = {c.strip() for c in new.split(",") if c.strip()}
            save_connectors_to_csv(connectors)
            st.success(f"{lvl} connectors updated.")

    # Student Codes Editor
    st.subheader("Student Codes")
    codes = load_student_codes()
    st.write(sorted(codes))
    add = st.text_area("Add codes (comma-separated):")
    if st.button("Add Codes"):
        for c in [x.strip() for x in add.split(",") if x.strip()]:
            codes.add(c)
        with open(STUDENT_CODES_PATH,"w",newline="",encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["student_code"])
            for c in sorted(codes): w.writerow([c])
        st.success("Student codes updated.")

    # Submission Log
    st.subheader("Submission Log")
    log = load_submission_log()
    df_log = pd.DataFrame(log.items(),columns=["Code","Count"])
    st.dataframe(df_log)
    st.download_button("Download Log", data=df_log.to_csv(index=False), file_name="submission_log.csv")

    # Collected Essays for Training
    st.subheader("Collected Essays for Training")
    download_training_data()
    st.stop()

# --- Student View ---
connectors_by_level = load_connectors_from_csv()
student_codes = load_student_codes()
log_data = load_submission_log()

level = st.selectbox("Select your level", ["A1","A2","B1","B2"])
tasks = ["Formal Letter","Informal Letter"] + (["Opinion Essay"] if level in ("B1","B2") else [])
task_type = st.selectbox("Select task type", tasks)

st.markdown("### ✍️ Structure & Tips")
with st.expander("✍️ Writing Tips"):
    if level=="A1":
        st.markdown("- Ich bin… / Ich habe…\n- Kurze Sätze\n- und, aber, weil\n- Großschreibung von Nomen")
    elif level=="A2":
        st.markdown("- Gründe mit weil/denn\n- Zeitangaben (z.B. am Montag)\n- Höflichkeitsformen")
    elif level=="B1":
        st.markdown("- Pro-und-Kontra\n- einerseits…andererseits\n- Nebensätze")
    else:
        st.markdown("- Passiv, indirekte Rede\n- Relativ- & Konditionalsätze")

student_id = st.text_input("Enter your student code:")
if not student_id:
    st.warning("Please enter your student code."); st.stop()
if student_id not in student_codes:
    st.error("❌ Unauthorized"); st.stop()

subs = log_data.get(student_id,0)
max_subs = 40 if level=="A1" else 45
if subs>=max_subs:
    st.warning(f"⚠️ You have reached {max_subs} submissions."); st.stop()
if subs>=max_subs-12:
    st.info("⏳ Few submissions left—be careful!")

task_num = None
task = None
if level=="A1":
    task_num = st.number_input(f"Aufgabe (1–{len(a1_tasks)})",1,len(a1_tasks),1)
    task = a1_tasks[task_num]
    st.markdown(f"### Aufgabe {task_num}: {task['task']}")
    for p in task["points"]:
        st.markdown(f"- {p}")

with st.form("f"):
    text = st.text_area("✏️ Your text:", height=300)
    submit = st.form_submit_button("Submit")

def score_text(txt, lvl, gpt_res, adv):
    ws = re.findall(r"\w+", txt.lower())
    uniq = len(set(ws))/len(ws) if ws else 0
    sents = re.split(r"[.!?]", txt)
    avg = len(ws)/max(1,len([s for s in sents if s.strip()]))
    read = "Easy" if avg<=12 else "Medium" if avg<=17 else "Hard"
    content=10
    grammar=max(1,5-len(gpt_res))
    vocab=min(5,int((len(set(ws))/len(ws))*5)) if ws else 1
    if adv: vocab=max(1,vocab-1)
    struct=5
    total=content+grammar+vocab+struct
    return content,grammar,vocab,struct,total,uniq,avg,read

def generate_feedback_text(level, task_type, task, c, g, v, s, tot, gpt_res, adv, used, txt):
    return (
        f"Your Feedback – {task_type} ({level})\n"
        f"Task: {task['task'] if task else ''}\n"
        f"- Content: {c}/10\n- Grammar: {g}/5\n- Vocabulary: {v}/5\n- Structure: {s}/5\n"
        f"Total: {tot}/25\n\n"
        "Grammar Suggestions:\n" + ("\n".join(gpt_res) if gpt_res else "None") + "\n\n"
        "Advanced Vocabulary:\n" + (", ".join(adv) if adv else "None") + "\n\n"
        "Connectors Used:\n" + (", ".join(used) if used else "None") + "\n\n"
        "Your Text:\n" + txt
    )

if submit:
    if not text.strip():
        st.warning("Enter text"); st.stop()
    with st.spinner("Processing…"):
        gpt_res = grammar_check_with_gpt(text)
        adv = detect_advanced_vocab(text, level) if level in ("A1","A2") else []
        c,g,v,s,tot,uniq,avg,read = score_text(text, level, gpt_res, adv)

        # update log & save training
        log_data[student_id] = subs+1
        with open(LOG_PATH,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            for k,vv in log_data.items(): w.writerow([k,vv])
        save_for_training(student_id, level, task_type, task_num, text, gpt_res, "")

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

        st.download_button("💾 Download feedback", data=feedback_text, file_name="feedback.txt")
