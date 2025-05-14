import re
from collections import Counter
import csv
import os
import streamlit as st
import pandas as pd
import openai

# ✅ App by Learn Language Education Academy
st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")
st.title("📝 German Letter & Essay Checker – Learn Language Education Academy")

# --- Teacher Settings ---
st.sidebar.header("🔧 Teacher Settings")
teacher_password = st.sidebar.text_input("🔒 Enter teacher password", type="password")
teacher_mode = (teacher_password == "Felix029")
if teacher_mode:
    page = st.sidebar.radio("Go to:", ["Student View", "Teacher Dashboard"])
else:
    page = "Student View"

# --- Secure API key retrieval ---
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as environment variable.")
    st.stop()
openai.api_key = api_key
client = openai.OpenAI(api_key=api_key)

# --- Load student codes ---
student_codes = set()
if os.path.exists("student_codes.csv"):
    with open("student_codes.csv", newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        idx = headers.index('student_code') if headers and 'student_code' in headers else 0
        for row in reader:
            if len(row) > idx and row[idx].strip():
                student_codes.add(row[idx].strip())

# --- Load submission log ---
log_path = "submission_log.csv"
log_data = {}
if os.path.exists(log_path):
    with open(log_path, newline='', encoding="utf-8") as f:
        for sid, count in csv.reader(f):
            log_data[sid] = int(count)

# --- Approved vocabulary by level ---
approved_vocab = {
    "A1": {"Anfrage","Anmelden","Terminen","Preisen","Kreditkarte","absagen",
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
           "schreiben","lesen","sehen","hören"},
    "A2": {"verstehen","arbeiten","lernen","besuchen","fahren","lesen",
           "helfen","sprechen","finden","tragen","essen","geben",
           "wohnen","spielen","anmelden","krankenhaus","trainingszeiten",
           "kosten","Termin","Ausweis","Führerschein","Öffnungszeiten",
           "verabreden","verschieben","absagen","einladen","Reparatur",
           "Schlüssel","Nachricht","E-Mail","Reise","Urlaub","Hotel",
           "Bahnhof","Flughafen","schmecken","bestellen","bezahlen",
           "trinken","kochen","Kollege","Chef","Arbeit","Stelle","Firma"}
}

# --- GPT-based grammar check ---
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

# --- Detect advanced vocabulary ---
def detect_advanced_vocab(text: str, level: str) -> list[str]:
    allowed = approved_vocab.get(level, set())
    words = re.findall(r"\w+", text)
    return [w for w in set(words) if w not in allowed]

# --- Approved Connectors Section ---
if teacher_mode and page == "Teacher Dashboard":
    st.subheader("Approved Connectors by Level")
    connector_path = "approved_connectors.csv"
    if os.path.exists(connector_path):
        approved_connectors = {}
        with open(connector_path, newline='', encoding="utf-8") as f:
            reader = csv.reader(f)
            for lvl, conns in reader:
                approved_connectors[lvl] = set(c.strip() for c in conns.split(',') if c.strip())
    else:
        approved_connectors = {
            "A1": {"weil","denn","ich möchte wissen","deshalb"},
            "A2": {"deshalb","deswegen","darum","trotzdem","obwohl","sobald","außerdem","zum Beispiel","und","aber","oder","erstens","zweitens","zum Schluss"},
            "B1": {"jedoch","allerdings","hingegen","trotzdem","dennoch","folglich","daher","demnach","infolgedenden","deshalb","damit","sofern","falls","währenddessen","inzwischen","mittlerweile","anschließend","schließlich","beispielsweise","zumal","wohingegen","erstens","zweitens","kurzum","zusammenfassend","einerseits","andererseits"},
            "B2": {"allerdings","dennoch","gleichwohl","demzufolge","mithin","ergo","sodass","obgleich","obschon","wenngleich","ungeachtet","indessen","nichtsdestotrotz","einerseits","andererseits","zumal","insofern","insoweit","demgemäß","zusammenfassend","abschließend","letztendlich"}
        }
    # Editable by teacher only
    for lvl in ["A1","A2","B1","B2"]:
        current = ", ".join(sorted(approved_connectors.get(lvl, [])))
        updated = st.text_area(f"{lvl} connectors (editable):", current, key=f"conn_{lvl}")
        if st.button(f"Update {lvl} connectors", key=f"btn_conn_{lvl}"):
            approved_connectors[lvl] = set(c.strip() for c in updated.split(',') if c.strip())
            with open(connector_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                for level_key, conns in approved_connectors.items():
                    writer.writerow([level_key, ",".join(sorted(conns))])
            st.success(f"✅ {lvl} connectors updated.")
    connectors_by_level = approved_connectors
else:
    # student view: just load connectors silently
    if os.path.exists("approved_connectors.csv"):
        approved_connectors = {}
        with open("approved_connectors.csv", newline='', encoding="utf-8") as f:
            for lvl, conns in csv.reader(f):
                approved_connectors[lvl] = set(c.strip() for c in conns.split(',') if c.strip())
    else:
        # use defaults
        approved_connectors = {
            "A1": {"weil","denn","ich möchte wissen","deshalb"},
            "A2": {"deshalb","deswegen","darum","trotzdem","obwohl","sobald","außerdem","zum Beispiel","und","aber","oder","erstens","zweitens","zum Schluss"},
            "B1": {"jedoch","allerdings","hingegen","trotzdem","dennoch","folglich","daher","demnach","infolgedenden","deshalb","damit","sofern","falls","währenddessen","inzwischen","mittlerweile","anschließend","schließlich","beispielsweise","zumal","wohingegen","erstens","zweitens","kurzum","zusammenfassend","einerseits","andererseits"},
            "B2": {"allerdings","dennoch","gleichwohl","demzufolge","mithin","ergo","sodass","obgleich","obschon","wenngleich","ungeachtet","indessen","nichtsdestotrotz","einerseits","andererseits","zumal","insofern","insoweit","demgemäß","zusammenfassend","abschließend","letztendlich"}
        }
    connectors_by_level = approved_connectors

# --- A1 Schreiben Tasks ---
a1_tasks = {
    1: {"task": "E-Mail an Ihren Arzt: Termin absagen.",        "points": ["Warum schreiben Sie?","Grund für die Absage.","Fragen Sie nach neuem Termin."]},
    2: {"task": "Einladung an Freund: Feier neuen Jobs.",        "points": ["Warum?","Wann?","Wer soll was mitbringen?"]},
    3: {"task": "E-Mail an Freund: Besuch ankündigen.",          "points": ["Warum?","Wann?","Was zusammen machen?"]},
    4: {"task": "E-Mail an Schule: Deutschkurs anfragen.",      "points": ["Warum?","Was möchten Sie wissen?","Wie antworten sie?"]},
    5: {"task": "E-Mail an Vermieterin: Heizung defekt.",       "points": ["Warum?","Seit wann?","Was soll sie tun?"]},
    6: {"task": "E-Mail an Freund: neue Wohnung.",              "points": ["Warum?","Wo ist sie?","Was gefällt Ihnen?"]},
    7: {"task": "E-Mail an Freundin: neue Arbeitsstelle.",      "points": ["Warum?","Wo?","Was machen Sie?"]},
    8: {"task": "E-Mail an Lehrer: Kurs nicht teilnehmen.",     "points": ["Warum?","Warum kommen Sie nicht?","Was möchten Sie?"]},
    9: {"task": "E-Mail an Bibliothek: Buch verloren.",         "points": ["Warum?","Welches Buch?","Was möchten Sie?"]},
    10: {"task": "E-Mail an Freundin: Urlaub planen.",          "points": ["Wohin?","Was machen?","Wann?"]},
    11: {"task": "E-Mail an Schule: Termin ändern.",           "points": ["Welcher Termin?","Wann haben Sie Zeit?","Warum?"]},
    12: {"task": "E-Mail an Bruder: Party organisieren.",       "points": ["Wann?","Was soll er mitbringen?","Warum?"]},
    13: {"task": "E-Mail an Freundin: Sie sind krank.",          "points": ["Warum?","Was machen Sie nicht?","Was sollen Sie tun?"]},
    14: {"task": "E-Mail an Nachbarn: Urlaub.",                "points": ["Wie lange?","Was sollen Nachbarn?","Warum informieren?"]},
    15: {"task": "Deutschlehrerin: Prüfung anmelden.",          "points": ["Welche Prüfung?","Warum?","Wann?"]},
    16: {"task": "Freundin: neuen Computer kaufen.",            "points": ["Warum?","Wo gekauft?","Was gefällt?"]},
    17: {"task": "Freundin: zusammen Sport.",                  "points": ["Welchen Sport?","Wann?","Warum?"]},
    18: {"task": "Freund: Hilfe Umzug.",                        "points": ["Wann?","Was soll er tun?","Warum?"]},
    19: {"task": "Freundin: Fest organisieren.",               "points": ["Wo?","Was machen?","Warum?"]},
    20: {"task": "Freundin: zusammen kochen.",                "points": ["Was wollen Sie kochen?","Wann?","Warum?"]},
    21: {"task": "Freund: neuer Job.",                         "points": ["Wo?","Was machen?","Warum?"]},
    22: {"task": "E-Mail an Schule: Deutschkurs besuchen.",     "points": ["Wann?","Warum?","Was möchten Sie?"]}
}

# --- Teacher Dashboard ---
if teacher_mode and page == "Teacher Dashboard":
    st.header("📊 Teacher Dashboard")
    st.subheader("Student Codes")
    st.write(sorted(student_codes))
    new_codes = st.text_area("Add student codes (comma-separated):")
    if st.button("Add to Student Codes"):
        for code in [s.strip() for s in new_codes.split(',') if s.strip()]:
            student_codes.add(code)
        with open("student_codes.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["student_code"])
            for c in sorted(student_codes):
                writer.writerow([c])
        st.success("✅ Student codes updated.")
    st.subheader("Submission Log")
    df = pd.DataFrame(list(log_data.items()), columns=["Student Code","Submissions"])
    st.dataframe(df)
    st.download_button("💾 Download Log", data=df.to_csv(index=False).encode('utf-8'), file_name="submission_log.csv", mime='text/csv')
    st.stop()

# --- Student View ---
level = st.selectbox("Select your level", ["A1","A2","B1","B2"])
tasks = ["Formal Letter","Informal Letter"]
if level in ("B1","B2"): tasks.append("Opinion Essay")
task_type = st.selectbox("Select task type", tasks)

# Writing Tips
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

# Authentication
student_id = st.text_input("Enter your student code:")
if not student_id:
    st.warning("Please enter your student code.")
    st.stop()
if student_id not in student_codes:
    st.error("❌ You are not authorized to use this app.")
    st.stop()

# Submission count check
subs = log_data.get(student_id, 0)
max_subs = 40 if level == 'A1' else 45
if subs >= max_subs:
    st.warning(f"⚠️ You have reached the maximum of {max_subs} submissions.")
    st.stop()
if subs >= max_subs - 12:
    st.info("⏳ You have used most of your submission chances. Review carefully!")

# Display A1 task
if level == "A1":
    task_num = st.number_input(f"Choose a Schreiben task number (1 to {len(a1_tasks)})", 1, len(a1_tasks), 1)
    task = a1_tasks[task_num]
    st.markdown(f"### Aufgabe {task_num}: {task['task']}")
    st.markdown("**Points:**")
    for p in task['points']:
        st.markdown(f"- {p}")

# Submission form
with st.form("feedback_form"):
    student_text = st.text_area("✏️ Write your letter or essay below:", height=300)
    submit = st.form_submit_button("✅ Submit for Feedback")

if submit:
    if not student_text.strip():
        st.warning("Please enter your text before submitting.")
        st.stop()
    subs += 1
    log_data[student_id] = subs
    with open(log_path, "w", newline='', encoding="utf-8") as f:
        csv.writer(f).writerows(log_data.items())

    # Grammar & vocab
    gpt_results = grammar_check_with_gpt(student_text)
    adv = detect_advanced_vocab(student_text, level) if level in ("A1","A2") else []
    if adv:
        st.warning(f"⚠️ The following words may be too advanced: {', '.join(adv)}")

    # Readability
    words = re.findall(r"\w+", student_text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    sentences = re.split(r'[.!?]', student_text)
    avg_words = len(words) / max(1, len([s for s in sentences if s.strip()]))
    readability = "Easy" if avg_words <= 12 else "Medium" if avg_words <= 17 else "Hard"
    st.markdown(f"🧮 Readability: {readability} ({avg_words:.1f} w/s)")

    # Scoring
    content_score = 10
    grammar_score = max(1, 5 - len(gpt_results))
    vocab_score = min(5, int((len(set(words))/len(words))*5))
    if adv:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score

# Display scores
    st.metric("Content", f"{content_score}/10")
    st.metric("Grammar", f"{grammar_score}/5")
    st.metric("Vocabulary", f"{vocab_score}/5")
    st.metric("Structure", f"{structure_score}/5")
    st.markdown(f"**Total: {total}/25**")

    # Why these scores explanation
    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: ratio {unique_ratio:.2f}, penalties ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    # Grammar suggestions
    if gpt_results:
        st.markdown("**Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

    # Connector suggestions for students only
    if not teacher_mode:
        # show a small subset of connectors as hints
        hints = sorted(connectors_by_level.get(level, []))[:4]
        st.info(f"📝 Try connectors like: {', '.join(hints)}…")

    # Full connector feedback (teacher or detailed view)
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

    # Download feedback
    st.download_button("💾 Download feedback", data="", file_name="feedback.txt")