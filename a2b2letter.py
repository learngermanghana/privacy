import re
from collections import Counter
import json
import os

import streamlit as st
import openai

# --- Streamlit config ---
st.set_page_config(
    page_title="German Letter & Essay Checker",
    layout="wide"
)

# --- Teacher Settings ---
st.sidebar.header("Teacher Settings")

# 1) Let teacher optionally enter their OpenAI key
api_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    help="Enter your OpenAI API key here (or set it in Streamlit secrets or as env var)."
)

# 2) Try secrets → env var → sidebar input
openai.api_key = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or api_input
)

if not openai.api_key:
    st.error(
        "❌ OpenAI API key not found. Please enter it above, "
        "add it to Streamlit secrets (key: OPENAI_API_KEY), or set the "
        "OPENAI_API_KEY environment variable."
    )
    st.stop()

# Custom phrases
st.sidebar.markdown(
    "Custom advanced (phrase;replacement) and forbidden phrases for your course."
)
custom_adv = st.sidebar.text_area(
    "Custom advanced (phrase;replacement)", height=100
)
custom_forbidden = st.sidebar.text_area(
    "Custom forbidden phrases", height=100
)

# Valid student IDs (one per line)
allowed_ids_text = st.sidebar.text_area(
    "Valid student IDs (one per line)", height=100,
    help="Enter the list of codes you issued to students"
)
allowed_ids = [s.strip() for s in allowed_ids_text.splitlines() if s.strip()]

# Max submissions per student
max_sub = st.sidebar.number_input(
    "Max submissions per student this session", min_value=1, max_value=20, value=5,
    key="max_sub"
)

# --- Helper Functions ---
def grammar_check_with_gpt(text: str):
    prompt = (
        "You are a German language tutor. "
        "Check the following German text for grammar and spelling errors. "
        "For each error, return a line in this format:\n"
        "`<error substring>` ⇒ `<suggestion>` — `<brief English explanation>`\n\n"
        f"Text:\n{text}"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().splitlines()

def detect_advanced_vocab(text: str, level: str):
    prompt = f"""
You are a German language expert. Identify any words in the following German text that exceed the {level} vocabulary level.
Respond in JSON format: {{"advanced": ["word1","word2",...]}}

Text:
{text}
"""
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        data = json.loads(resp.choices[0].message.content)
        return data.get("advanced", [])
    except Exception:
        return []

# Context-Sensitive Tips
TIPS = {
    'A2_Formal Letter': [
        'Begin with: "Sehr geehrte Damen und Herren,"',
        'Einleitung: Begründen Sie kurz Ihr Anliegen.',
        'Hauptteil: Führen Sie Details aus.',
        'Schlussteil: Bitten Sie um Rückmeldung.',
        'Abschluss: "Mit freundlichen Grüßen"'
    ],
    'A2_Informal Letter': [
        'Begin with: "Liebe/r ...,"',
        'Einleitung: Frage, wie es der Person geht.',
        'Hauptteil: Erzählen Sie Ihr Anliegen.',
        'Schlussteil: Bitte um Antwort.',
        'Abschluss: "Liebe Grüße"'
    ],
    'B1_Opinion Essay': [
        'Einleitung: Thema vorstellen und eigene Meinung ankündigen.',
        'Hauptteil: Pro-Argumente, Kontra-Argumente.',
        'Schlussteil: Eigene Stellungnahme.',
        'Abschluss: Zusammenfassen und Ausblick.'
    ]
}

# Recommended Structure Slide
st.markdown("## ✏️ Recommended Writing Structure")
sections = [
    ("Introduction", "Introduce your topic and state your purpose or opinion."),
    ("Body", "Provide supporting arguments, examples, or details in clear paragraphs."),
    ("Conclusion", "Summarize your main points and, if appropriate, include a closing remark.")
]
sel = st.select_slider(
    "Navigate sections:",
    options=[sec[0] for sec in sections],
    value=sections[0][0]
)
for title, desc in sections:
    if sel == title:
        st.markdown(f"**{title}:** {desc}")
        break

# --- Main UI ---
st.title("📄 German Letter & Essay Checker")
st.subheader("By Learn Language Education Academy")
st.markdown("### ✍️ Structure & Tips")

# Student selects their level and task type
level = st.selectbox("Select your level", ["A2", "B1", "B2"])
tasks = ["Formal Letter", "Informal Letter"]
if level in ("B1", "B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select your task type", tasks)

# Require unique student ID
student_id = st.text_input(
    "Enter your student ID (provided by your teacher):", value="",
    key="student_id"
)
if not student_id:
    st.warning("Please enter your student ID before submitting.")
    st.stop()
if allowed_ids and student_id not in allowed_ids:
    st.error("❌ Invalid student ID. Please use the code provided by your teacher.")
    st.stop()

# Initialize submission counter per student_id
sess_key = f"count_{student_id}"
if sess_key not in st.session_state:
    st.session_state[sess_key] = 0

with st.form("feedback_form"):
    if st.session_state[sess_key] >= max_sub:
        st.warning(f"⚠️ You have reached the limit of {max_sub} submissions this session.")
        submit = False
    else:
        student_letter = st.text_area("Write your letter or essay below:", height=350)
        submit = st.form_submit_button("✅ Submit for Feedback")

if submit:
    st.session_state[sess_key] += 1
    text = student_letter.strip()
    if not text:
        st.warning("Please enter your text before submitting.")
        st.stop()

    # Grammar check
    with st.spinner("Checking with GPT…"):
        try:
            gpt_results = grammar_check_with_gpt(text)
        except Exception as e:
            st.error(f"GPT check failed: {e}")
            gpt_results = []

    # Advanced vocab for A2
    if level == 'A2':
        with st.spinner("Checking for advanced vocabulary…"):
            advanced_words = detect_advanced_vocab(text, level)
        if advanced_words:
            sample = ', '.join(advanced_words[:5])
            st.warning(f"⚠️ Detected advanced vocabulary beyond {level} level: {sample}{'...' if len(advanced_words)>5 else ''}")

    # Vocabulary metrics
    words = re.findall(r"\w+", text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    counts = Counter(words)
    repeated = [w for w, c in counts.items() if c > 3]
    repeat_penalty = sum(c - 3 for c in counts.values() if c > 3)

    # Readability metrics
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
    if avg_words_per_sentence <= 12:
        readability = "Easy"
    elif avg_words_per_sentence <= 17:
        readability = "Medium"
    else:
        readability = "Hard"
    st.markdown(f"🧮 Readability: {readability} ({avg_words_per_sentence:.1f} words/sentence)")

    # Scoring rubric
    content_score = 10
    grammar_score = max(1, 5 - len(gpt_results))
    vocab_score = min(5, int(unique_ratio * 5))
    vocab_score = max(1, vocab_score - repeat_penalty)
    if repeated:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score

    # Display breakdown with emojis
    colors = {'Content': '#4e79a7', 'Grammar': '#e15759', 'Vocabulary': '#76b7b2', 'Structure': '#59a14f'}
    st.markdown(f"<span style='color:{colors['Content']}'>📖 Content: {content_score}/10</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Grammar']}'>✏️ Grammar: {grammar_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Vocabulary']}'>💬 Vocabulary: {vocab_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Structure']}'>🔧 Structure: {structure_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"🏆 **Total: {total}/25**")

    # Explanation
    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: {len(gpt_results)} errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: ratio {unique_ratio:.2f}, penalties ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    # Pass threshold
    threshold = 18 if level == 'A2' else 20
    pass_msg = (
        "🎉 You passed! Send this to your tutor for final review." if total >= threshold else
        "⚠️ Below pass mark. Review feedback or contact your tutor."
    )
    if total >= threshold:
        st.info(pass_msg)
    else:
        st.warning(pass_msg)

    # GPT suggestions
    if gpt_results:
        st.markdown("**GPT Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

        # Annotated text
        ann = text
        for line in gpt_results:
            err = line.split("⇒")[0].strip(" `")
            ann = re.sub(
                re.escape(err),
                f"<span style='background-color:{colors['Grammar']}; color:#fff'>{err}</span>",
                ann,
                flags=re.I
            )

    # Annotated text rendering (escape newline properly)
    safe_ann = ann.replace("\n", "  \n")
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(safe_ann, unsafe_allow_html=True)

    # Download feedback (plain text)
    feedback_lines = "\n".join(gpt_results)
    feedback_txt = f"Score: {total}/25\n{feedback_lines}"

    st.download_button(
        label="💾 Download feedback",
        data=feedback_txt,
        file_name="feedback.txt"
    )
