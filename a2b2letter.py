import re
from collections import Counter
import json
import os
import streamlit as st
import openai
import csv

# ✅ App by Learn Language Education Academy

import streamlit as st
st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")
st.title("📝 German Letter & Essay Checker – Learn Language Education Academy")

# --- Teacher Settings ---
st.sidebar.header("Teacher Settings")
teacher_password = st.sidebar.text_input("🔒 Enter teacher password", type="password")
teacher_mode = teacher_password == "Felix029"
if teacher_mode:
    st.sidebar.markdown("**🔗 Connector Words by Level:**")
    connector_input = st.sidebar.text_area("Enter connector words (comma-separated):", "und, aber, weil, denn")
    st.sidebar.success("✅ Teacher Mode Enabled")
    st.sidebar.markdown("**🔧 Suggested Phrases or Ideas:**")
    teacher_ideas = st.sidebar.text_area("Add common sentence starters or vocabulary for students:")

# ✅ Secure API key retrieval
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as an environment variable.")
    st.stop()

openai.api_key = api_key

# --- GPT-based grammar check ---
def grammar_check_with_gpt(text: str):
    prompt = (
        "You are a German language tutor. "
        "Check the following German text for grammar and spelling errors. "
        "For each error, return a line in this format:\n"
        "`<error substring>` ⇒ `<suggestion>` — `<brief English explanation>`\n\n"
        f"Text:\n{text}"
    )
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip().splitlines()

# --- GPT-based vocabulary difficulty check ---
def detect_advanced_vocab(text: str, level: str):
    allowed_a1_words = set([
        "Anfrage", "Anmelden", "Terminen", "Preisen", "Kreditkarte", "absagen", 
        "anfangen", "vereinbaren", "übernachten", "Rechnung", "Informationen",
        "Anruf", "antworten", "Gebühr", "buchen", "eintragen", "mitnehmen",
        "Unterschrift", "Untersuchung", "Unfall", "abholen", "abgeben",
        "mitteilen", "erreichen", "eröffnen", "reservieren", "verschieben"
    ])
    prompt = f"""
You are a German language expert. Identify any words in the following German text that exceed the {level} vocabulary level.
Respond in JSON format: {{"advanced": ["word1","word2",...]}}
Text:
{text}
"""
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        data = json.loads(response.choices[0].message.content)
        return [word for word in data.get("advanced", []) if word not in allowed_a1_words or level != "A1"]
    except Exception:
        return []

# --- A1 Schreiben Tasks ---
a1_tasks = {
    1: {"task": "Schreiben Sie eine E-Mail an Ihren Arzt und sagen Sie Ihren Termin ab.", "points": ["Warum schreiben Sie?", "Sagen Sie: den Grund für die Absage.", "Fragen Sie: nach einem neuen Termin."]},
    2: {"task": "Schreiben Sie eine Einladung an Ihren Freund zur Feier Ihres neuen Jobs.", "points": ["Warum schreiben Sie?", "Wann ist die Feier?", "Wer soll was mitbringen?"]},
    3: {"task": "Schreiben Sie eine E-Mail an einen Freund und teilen Sie ihm mit, dass Sie ihn besuchen möchten.", "points": ["Warum schreiben Sie?", "Wann besuchen Sie ihn?", "Was möchten Sie zusammen machen?"]},
    4: {"task": "Schreiben Sie eine E-Mail an Ihre Schule und fragen Sie nach einem Deutschkurs.", "points": ["Warum schreiben Sie?", "Was möchten Sie wissen?", "Wie kann die Schule antworten?"]},
    5: {"task": "Schreiben Sie eine E-Mail an Ihre Vermieterin. Ihre Heizung ist kaputt.", "points": ["Warum schreiben Sie?", "Seit wann ist die Heizung kaputt?", "Was soll die Vermieterin tun?"]},
    6: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben eine neue Wohnung.", "points": ["Warum schreiben Sie?", "Wo ist die Wohnung?", "Was gefällt Ihnen?"]},
    7: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben eine neue Arbeitsstelle.", "points": ["Warum schreiben Sie?", "Wo arbeiten Sie jetzt?", "Was machen Sie?"]},
    8: {"task": "Schreiben Sie eine E-Mail an Ihren Lehrer. Sie können am Kurs nicht teilnehmen.", "points": ["Warum schreiben Sie?", "Warum kommen Sie nicht?", "Was möchten Sie?"]},
    9: {"task": "Schreiben Sie eine E-Mail an die Bibliothek. Sie haben ein Buch verloren.", "points": ["Warum schreiben Sie?", "Welches Buch haben Sie verloren?", "Was möchten Sie wissen?"]},
    10: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten mit ihr in den Urlaub fahren.", "points": ["Warum schreiben Sie?", "Wohin möchten Sie fahren?", "Was möchten Sie dort machen?"]},
    11: {"task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Termin ändern.", "points": ["Warum schreiben Sie?", "Welcher Termin ist es?", "Wann haben Sie Zeit?"]},
    12: {"task": "Schreiben Sie eine E-Mail an Ihren Bruder. Sie machen eine Party.", "points": ["Warum schreiben Sie?", "Wann ist die Party?", "Was soll Ihr Bruder mitbringen?"]},
    13: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie sind krank.", "points": ["Warum schreiben Sie?", "Was machen Sie heute nicht?", "Was sollen Sie tun?"]},
    14: {"task": "Schreiben Sie eine E-Mail an Ihre Nachbarn. Sie machen Urlaub.", "points": ["Warum schreiben Sie?", "Wie lange sind Sie weg?", "Was sollen die Nachbarn tun?"]},
    15: {"task": "Schreiben Sie eine E-Mail an Ihre Deutschlehrerin. Sie möchten eine Prüfung machen.", "points": ["Warum schreiben Sie?", "Welche Prüfung möchten Sie machen?", "Wann möchten Sie die Prüfung machen?"]},
    16: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie haben einen neuen Computer gekauft.", "points": ["Warum schreiben Sie?", "Wo haben Sie den Computer gekauft?", "Was gefällt Ihnen besonders?"]},
    17: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen Sport machen.", "points": ["Warum schreiben Sie?", "Welchen Sport möchten Sie machen?", "Wann können Sie?"]},
    18: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie brauchen Hilfe beim Umzug.", "points": ["Warum schreiben Sie?", "Wann ist der Umzug?", "Was soll Ihr Freund machen?"]},
    19: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten ein Fest organisieren.", "points": ["Warum schreiben Sie?", "Wo soll das Fest sein?", "Was möchten Sie machen?"]},
    20: {"task": "Schreiben Sie eine E-Mail an Ihre Freundin. Sie möchten zusammen kochen.", "points": ["Warum schreiben Sie?", "Was möchten Sie kochen?", "Wann möchten Sie kochen?"]},
    21: {"task": "Schreiben Sie eine E-Mail an Ihren Freund. Sie haben einen neuen Job.", "points": ["Warum schreiben Sie?", "Wo arbeiten Sie jetzt?", "Was machen Sie?"]},
    22: {"task": "Schreiben Sie eine E-Mail an Ihre Schule. Sie möchten einen Deutschkurs besuchen.", "points": ["Warum schreiben Sie?", "Wann möchten Sie den Kurs besuchen?", "Was möchten Sie noch wissen?"]}
}

# --- Level Selection and Task Type ---
level = st.selectbox("Select your level", ["A1", "A2", "B1", "B2"])
tasks = ["Formal Letter", "Informal Letter"]
if level in ("B1", "B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select your task type", tasks)


# --- Writing Tips and Usage Advice ---
st.markdown("### ✍️ Structure & Tips")
with st.expander("✍️ Show Writing Tips and Usage Advice"):
    if level == "A1":
        st.markdown("""✏️ **A1 Writing Tips:**
- Use simple present tense (ich bin, ich habe, ich wohne...)
- Keep sentences short and clear
- Use basic connectors like *und*, *aber*, *weil*
- Avoid complex verbs or modal structures
- Always start sentences with a capital letter

📌 **App Usage Advice:**
- Before clicking submit, take your time to read all the grammar and vocabulary suggestions carefully.
- Don't fix one correction and immediately resubmit — instead, review all the feedback, correct everything in one go, and then submit.
- This will save your submission chances and help you learn better.
- If you're confused about something, it's always okay to ask your tutor for help.
- Use the highlighted feedback and suggestions at the bottom of the page to guide your corrections.""")
    elif level == "A2":
        st.markdown("""✏️ **A2 Writing Tips:**
- Explain reasons using *weil* and *denn*
- Add time expressions (z.B. *am Montag*, *um 8 Uhr*)
- Include polite forms like *ich möchte*, *könnten Sie...?*""")
    elif level == "B1":
        st.markdown("""✏️ **B1 Writing Tips:**
- Include both pros and cons in essays
- Use connectors like *einerseits... andererseits*, *deshalb*, *trotzdem*
- Vary sentence structure with subordinates""")
    elif level == "B2":
        st.markdown("""✏️ **B2 Writing Tips:**
- Support opinions with examples and evidence
- Use passive voice and indirect speech where appropriate
- Include complex sentence structures with relative and conditional clauses""")

# --- Approved Students Check ---
approved_students = set()
if os.path.exists("approved_students.csv"):
    with open("approved_students.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        approved_students = {row["student_id"] for row in reader}

student_id = st.text_input("Enter your student code (given by your teacher):", value="", key="student_id")
if not student_id:
    st.warning("Please enter your student code before submitting.")
    st.stop()
if student_id not in approved_students:
    st.error("❌ You are not authorized to use this app. Contact your teacher.")
    st.stop()

# --- Submission Count Check and Soft Warning ---
log_path = "submission_log.csv"
log_data = {}
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) == 2:
                log_data[row[0]] = int(row[1])

submission_count = log_data.get(student_id, 0)
max_subs = 40 if level == 'A1' else 45
if submission_count >= max_subs:
    st.warning(f"⚠️ You have reached the maximum of {max_subs} submissions for your level.")
    st.stop()

if submission_count >= (max_subs - 5):
    st.info("⏳ You have used most of your submission chances. Please review your feedback carefully before submitting again.")

# --- Task Display and Form Submission ---
if level == "A1":
    task_number = st.number_input(f"Choose a Schreiben task number (1 to {len(a1_tasks)})", 1, len(a1_tasks), 1)
    selected_task = a1_tasks[task_number]
    st.markdown(f"### 📄 Aufgabe {task_number}: {selected_task['task']}")
    st.markdown("**Your points:**")
    for point in selected_task['points']:
        st.markdown(f"- {point}")

with st.form("feedback_form"):
    student_letter = st.text_area("✏️ Write your letter or essay below:", height=350)
    submit = st.form_submit_button("✅ Submit for Feedback")

if submit:
    text = student_letter.strip()
    if not text:
        st.warning("Please enter your text before submitting.")
        st.stop()

    submission_count += 1
    log_data[student_id] = submission_count
    try:
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            for sid, count in log_data.items():
                writer.writerow([sid, count])
    except Exception as e:
        st.warning(f"⚠️ Failed to save submission count: {e}")

    with st.spinner("Checking with GPT…"):
        try:
            gpt_results = grammar_check_with_gpt(text)
        except Exception as e:
            st.error(f"GPT check failed: {e}")
            gpt_results = []

    try:
        advanced_words = detect_advanced_vocab(text, level)
    except Exception as e:
        advanced_words = []
        st.warning(f"Vocabulary level check failed: {e}")

    if advanced_words:
        st.warning(f"⚠️ The following words may be too advanced for {level}-level writing: {', '.join(advanced_words)}")

    words = re.findall(r"\w+", text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    counts = Counter(words)
    repeated = [w for w, c in counts.items() if c > 3]
    repeat_penalty = sum(c - 3 for c in counts.values() if c > 3)

    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
    readability = "Easy" if avg_words_per_sentence <= 12 else "Medium" if avg_words_per_sentence <= 17 else "Hard"
    st.markdown(f"🧮 Readability: {readability} ({avg_words_per_sentence:.1f} words/sentence)")

    content_score = 10
    grammar_score = max(1, 5 - len(gpt_results))
    vocab_score = min(5, int(unique_ratio * 5))
    vocab_score = max(1, vocab_score - repeat_penalty)
    if repeated:
        vocab_score = max(1, vocab_score - 1)
    if advanced_words and level in ["A1", "A2"]:
        vocab_score = max(1, vocab_score - 1)

    connector_words = [w.strip() for w in connector_input.split(',')] if teacher_mode else ["und", "aber", "oder", "denn", "weil", "deshalb"]
    used_connectors = [w for w in connector_words if w in text.lower()]

    if used_connectors:
        st.success(f"✅ Good job! You used connector(s): {', '.join(used_connectors)}")
    else:
        st.info("📝 Try to include simple connectors like *und*, *aber*, or *weil* to link your ideas.")

    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score

    colors = {
        'Content': '#4e79a7',
        'Grammar': '#e15759',
        'Vocabulary': '#76b7b2',
        'Structure': '#59a14f',
        'Advanced': '#f1c232'
    }

    st.markdown(f"<span style='color:{colors['Content']}'>📖 Content: {content_score}/10</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Grammar']}'>✏️ Grammar: {grammar_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Vocabulary']}'>💬 Vocabulary: {vocab_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Structure']}'>🔧 Structure: {structure_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"🏆 **Total: {total}/25**")

    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: ratio {unique_ratio:.2f}, penalties ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    threshold = 16 if level == 'A1' else 18 if level == 'A2' else 20
    if total >= threshold:
        st.info("🎉 You passed! Send this to your tutor for final review.")
    else:
        st.warning("⚠️ Below pass mark. Review feedback or contact your tutor.")

    if gpt_results:
        st.markdown("**GPT Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

    ann = text
    for line in gpt_results:
        err = line.split("⇒")[0].strip(" `")
        ann = re.sub(
            re.escape(err),
            f"<span style='background-color:{colors['Grammar']}; color:#fff'>{err}</span>",
            ann,
            flags=re.I
        )

    if advanced_words:
        for adv in advanced_words:
            ann = re.sub(
                rf'\b{re.escape(adv)}\b',
                f"<span title='Too advanced for {level}' style='background-color:{colors['Advanced']}; color:#000'>{adv}</span>",
                ann,
                flags=re.I
            )
        st.markdown("**📚 Advanced Vocabulary Used:**")
        for word in advanced_words:
            st.markdown(f"- {word} _(not recommended for {level})_")

    safe_ann = ann.replace("\n", "  \n")
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(safe_ann, unsafe_allow_html=True)

    feedback_txt = f"Score: {total}/25\n" + "\n".join(gpt_results)
    st.download_button(
        label="💾 Download feedback",
        data=feedback_txt,
        file_name="feedback.txt"
    )
