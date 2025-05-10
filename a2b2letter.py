import re
from collections import Counter

import streamlit as st
import openai

# --- Streamlit config ---
st.set_page_config(
    page_title="German Letter & Essay Checker",
    layout="wide"
)

# --- Teacher Settings ---
st.sidebar.header("Teacher Settings")
# Always use OpenAI for grammar & spelling
openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]

st.sidebar.markdown(
    "Custom advanced (phrase;replacement) and forbidden phrases for your course."
)
custom_adv = st.sidebar.text_area(
    "Custom advanced (phrase;replacement)", height=100
)
custom_forbidden = st.sidebar.text_area(
    "Custom forbidden phrases", height=100
)

# --- GPT Grammar Checker ---
def grammar_check_with_gpt(text: str):
    prompt = (
        "You are a German language tutor. "
        "Check the following German text for grammar and spelling errors. "
        "For each error, return a line in this format:\n"
        "`<error substring>` ⇒ `<suggestion>` — `<brief English explanation>`\n\n"
        f"Text:\n{text}"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # using cost-effective GPT-3.5 Turbo
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return resp.choices[0].message.content.strip().splitlines()

# --- Advanced Vocabulary Detection ---
def detect_advanced_vocab(text: str, level: str):
    prompt = (
        f"You are a German language expert. Identify any words in the following German text that exceed the {level} vocabulary level. "
        "Respond in JSON format: {\"advanced\": [list of words]}\n\n"
        f"Text:\n{text}"
    )
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    try:
        content = resp.choices[0].message.content.strip()
        data = __import__('json').loads(content)
        return data.get("advanced", [])
    except Exception:
        return []

# --- Context-Sensitive Tips ---
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

# --- Recommended Structure Slide ---
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

student_name = st.text_input("Enter your name:", value="Student")
level = st.selectbox("Select your level", ["A2", "B1", "B2"])

tasks = ["Formal Letter", "Informal Letter"]
if level in ("B1", "B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select your task type", tasks)

key = f"{level}_{task_type.replace(' ', '_')}"
if key in TIPS:
    st.markdown("**Tips:**")
    for tip in TIPS[key]:
        st.markdown(f"- {tip}")

with st.form("feedback_form"):
    student_letter = st.text_area("Write your letter or essay below:", height=350)
    submit = st.form_submit_button("✅ Submit for Feedback")

if submit:
    text = student_letter.strip()
    if not text:
        st.warning("Please enter your text before submitting.")
        st.stop()

    # 1) Grammar check (always on)
    with st.spinner("Checking with GPT…"):
        try:
            gpt_results = grammar_check_with_gpt(text)
        except Exception as e:
            st.error(f"GPT check failed: {e}")
            gpt_results = []

    # 1.2) Advanced vocabulary detection for level
    if level == 'A2':
        with st.spinner("Checking for advanced vocabulary…"):
            advanced_words = detect_advanced_vocab(text, level)
        if advanced_words:
            sample = ', '.join(advanced_words[:5])
            st.warning(f"⚠️ Detected advanced vocabulary beyond {level} level: {sample}{'...' if len(advanced_words)>5 else ''}")

    # 2) Vocabulary metrics
    words = re.findall(r"\w+", text.lower())
    unique_ratio = len(set(words)) / len(words) if words else 0
    counts = Counter(words)
    repeated = [w for w, c in counts.items() if c > 3]
    repeat_penalty = sum(c - 3 for c in counts.values() if c > 3)

    # 2.1) Readability metrics
    sentences = [s for s in re.split(r'[.!?]', text) if s.strip()]
    avg_words_per_sentence = len(words) / len(sentences) if sentences else 0
    if avg_words_per_sentence <= 12:
        readability = "Easy"
    elif avg_words_per_sentence <= 17:
        readability = "Medium"
    else:
        readability = "Hard"
    # Display readability
    st.markdown(f"🧮 Readability: {readability} ({avg_words_per_sentence:.1f} words/sentence)")

    # 3) Scoring rubric
    content_score = 10
    grammar_score = max(1, 5 - len(gpt_results))
    vocab_score = min(5, int(unique_ratio * 5))
    vocab_score = max(1, vocab_score - repeat_penalty)
    if repeated:
        vocab_score = max(1, vocab_score - 1)
    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score

    # 4) Display breakdown with emojis
    colors = {'Content':'#4e79a7','Grammar':'#e15759','Vocabulary':'#76b7b2','Structure':'#59a14f'}
    st.markdown(f"<span style='color:{colors['Content']}'>📖 Content: {content_score}/10</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Grammar']}'>✏️ Grammar: {grammar_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Vocabulary']}'>💬 Vocabulary: {vocab_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"<span style='color:{colors['Structure']}'>🔧 Structure: {structure_score}/5</span>", unsafe_allow_html=True)
    st.markdown(f"🏆 **Total: {total}/25**")

    # 5) Why these scores?
    st.markdown("**Why these scores?**")
    st.markdown(f"- 📖 Content: fixed = {content_score}/10")
    st.markdown(f"- ✏️ Grammar: 5 errors ⇒ {grammar_score}/5")
    st.markdown(f"- 💬 Vocabulary: ratio {unique_ratio:.2f}, penalties ⇒ {vocab_score}/5")
    st.markdown(f"- 🔧 Structure: fixed = {structure_score}/5")

    # 6) Determine pass threshold
    threshold = 18 if level == 'A2' else 20
    pass_msg = "🎉 You passed! Send this to your tutor for final review." if total >= threshold else "⚠️ Below pass mark. Review feedback or contact your tutor."
    if total >= threshold:
        st.info(pass_msg)
    else:
        st.warning(pass_msg)

    # 7) Show GPT suggestions
    if gpt_results:
        st.markdown("**GPT Grammar Suggestions:**")
        for line in gpt_results:
            st.markdown(f"- {line}")

    # 8) Annotated Text
    ann = text
    for line in gpt_results:
        err = line.split("⇒")[0].strip(" `")
        ann = re.sub(re.escape(err), f"<span style='background-color:{colors['Grammar']}; color:#fff'>{err}</span>", ann, flags=re.I)
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(ann.replace("\n", "  \n"), unsafe_allow_html=True)

    # 9) Download feedback
    feedback_txt = f"Score: {total}/25\n" + "\n".join(gpt_results)
    st.download_button(label="💾 Download feedback", data=feedback_txt, file_name="feedback.txt")
