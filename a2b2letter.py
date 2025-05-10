import re
from collections import Counter
import json
import os
import streamlit as st
import openai

# --- Streamlit config ---
st.set_page_config(page_title="German Letter & Essay Checker", layout="wide")

# --- Teacher Settings ---
st.sidebar.header("Teacher Settings")

# ✅ Secure API key retrieval
api_key = st.secrets.get("general", {}).get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OpenAI API key not found. Add it to secrets.toml under [general] or set as an environment variable.")
    st.stop()

# ✅ OpenAI configuration
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
        return data.get("advanced", [])
    except Exception:
        return []

# --- Main UI ---
st.title("📄 German Letter & Essay Checker")
st.subheader("By Learn Language Education Academy")
st.markdown("### ✍️ Structure & Tips")

level = st.selectbox("Select your level", ["A2", "B1", "B2"])
tasks = ["Formal Letter", "Informal Letter"]
if level in ("B1", "B2"):
    tasks.append("Opinion Essay")
task_type = st.selectbox("Select your task type", tasks)

student_id = st.text_input("Enter your student name (recognized by your teacher):", value="", key="student_id")
if not student_id:
    st.warning("Please enter your student name before submitting.")
    st.stop()

sess_key = f"count_{student_id}"
if sess_key not in st.session_state:
    st.session_state[sess_key] = 0

with st.form("feedback_form"):
    if st.session_state[sess_key] >= 5:
        st.warning("⚠️ You have reached the limit of 5 submissions this session.")
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

    with st.spinner("Checking with GPT…"):
        try:
            gpt_results = grammar_check_with_gpt(text)
        except Exception as e:
            st.error(f"GPT check failed: {e}")
            gpt_results = []

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
    structure_score = 5
    total = content_score + grammar_score + vocab_score + structure_score

    colors = {'Content': '#4e79a7', 'Grammar': '#e15759', 'Vocabulary': '#76b7b2', 'Structure': '#59a14f'}
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

    threshold = 18 if level == 'A2' else 20
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
    safe_ann = ann.replace("\n", "  \n")
    st.markdown("**Annotated Text:**", unsafe_allow_html=True)
    st.markdown(safe_ann, unsafe_allow_html=True)

    feedback_txt = f"Score: {total}/25\n" + "\n".join(gpt_results)
    st.download_button(
        label="💾 Download feedback",
        data=feedback_txt,
        file_name="feedback.txt"
    )
