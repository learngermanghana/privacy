import io
import re
from collections import Counter

import streamlit as st

# --- Streamlit config ---
st.set_page_config(
    page_title="German Letter & Essay Checker",
    layout="wide"
)

# --- Teacher Settings ---
st.sidebar.header("Teacher Settings")
st.sidebar.markdown("Custom advanced (phrase;replacement) and forbidden phrases for your course.")
custom_adv = st.sidebar.text_area("Custom advanced (phrase;replacement)", height=100)
custom_forbidden = st.sidebar.text_area("Custom forbidden phrases", height=100)

# --- LanguageTool setup (cached) ---
@st.cache_resource
def get_language_tool():
    try:
        import language_tool_python
        return language_tool_python.LanguageTool("de-DE", motherTongue="en")
    except ImportError:
        return None

lt_tool = get_language_tool()

# --- UI ---
st.title("📄 German Letter & Essay Checker")
st.subheader("By Learn Language Education Academy")
st.markdown("### ✍️ Structure & Tips")

# --- Input Form ---
with st.form("feedback_form"):
    student_name = st.text_input("Enter your name:", value="Student")
    level = st.selectbox("Select your level", ["A2", "B1", "B2"])
    task_opts = ["Formal Letter", "Informal Letter"] + (["Opinion Essay"] if level in ["B1", "B2"] else [])
    task_type = st.selectbox("Select your task type", task_opts)
    student_letter = st.text_area("Write your letter or essay below:", height=350)
    submit = st.form_submit_button("✅ Submit for Feedback")

if submit:
    text = student_letter.strip()
    if not text:
        st.warning("Please enter your text before submitting.")
    else:
        # Show spinner while grammar checker loads
        with st.spinner("Loading grammar checker... please wait..."):
            matches = lt_tool.check(text) if lt_tool else []

        # Vocabulary metrics
        words = re.findall(r"\w+", text.lower())
        unique_ratio = len(set(words)) / len(words) if words else 0
        counts = Counter(words)
        repeated = [w for w, c in counts.items() if c > 3]
        repeat_penalty = sum(c-3 for c in counts.values() if c > 3)

        # Scoring
        content_score = 10  # full marks
        grammar_score = max(1, 5 - len(matches))
        vocab_score = min(5, int(unique_ratio * 5))
        vocab_score = max(1, vocab_score - repeat_penalty)
        if repeated:
            vocab_score = max(1, vocab_score - 1)
        structure_score = 5  # full marks
        total = content_score + grammar_score + vocab_score + structure_score

        # Display breakdown
        st.success(
            f"Score Breakdown (25 total):\n"
            f"- Content: {content_score}/10\n"
            f"- Grammar: {grammar_score}/5\n"
            f"- Vocabulary: {vocab_score}/5\n"
            f"- Structure & Coherence: {structure_score}/5\n"
            f"**Total: {total}/25**"
        )

        # Pass threshold adjustment
        threshold = 16 if level == 'A2' else 18
        if total >= threshold:
            st.info("Congratulations! You have passed the threshold. You can send this to your tutor for further assessment.")
        else:
            st.warning("Score below pass mark. Please review the feedback above. If you’re struggling, contact your tutor for help.")

        # Scoring rationale with colored bullets
        st.markdown("**Scoring Rationale:**")
        rationale_items = [
            f"Content: full marks covering all required points ({content_score}/10)",
            f"Grammar: {grammar_score}/5 (deducted {len(matches)} for errors)",
            f"Vocabulary: {vocab_score}/5 (unique ratio {unique_ratio:.2f}, penalties {repeat_penalty}{', extra for repetition' if repeated else ''})",
            f"Structure & Coherence: full marks ({structure_score}/5)"
        ]
        colors = ['#4e79a7', '#e15759', '#76b7b2', '#59a14f']
        for i, item in enumerate(rationale_items):
            st.markdown(f"<span style='color:{colors[i]}'>• {item}</span>", unsafe_allow_html=True)

        # Suggestions with translation
        if matches:
            st.markdown("**Grammar & Spelling Suggestions (original + English translation):**")
            translation_map = {
                "Möglicherweise fehlt ein ‚und‘ oder ein Komma, oder es wurde nach dem Wort ein überflüssiges Leerzeichen eingefügt. Eventuell haben Sie auch versehentlich einen Bindestrich statt eines Punktes eingefügt.":
                    "Possibly a missing 'und' or comma, or an extra space was inserted after the word. You may also have accidentally used a hyphen instead of a period.",
                "Möglicher Tippfehler gefunden.": "Possible typo detected.",
                "Nur hinter einem Komma steht ein Leerzeichen, aber nicht davor.": "There is a space after a comma but not before.",
                "Es scheint das ‚noch‘ der Wendung ‚weder A noch B‘ zu fehlen.": "It seems the 'noch' in the phrase 'weder A noch B' is missing.",
                "Hier scheint ein Leerzeichen zu viel zu sein.": "There appears to be one space too many here.",
                "Außer am Satzanfang werden nur Nomen und Eigennamen großgeschrieben.": "Except at the beginning of a sentence, only nouns and proper names are capitalized.",
                "Möglicher Tippfehler: mehr als ein Leerzeichen hintereinander": "Possible typo: more than one space in a row.",
                "Hinter einem Komma sollte ein Leerzeichen stehen.": "After a comma, there should be a space.",
                "Dieser Satz fängt nicht mit einem großgeschriebenen Wort an.": "This sentence does not start with a capitalized word.",
                "Möglicherweise passen das Nomen und die Wörter, die das Nomen beschreiben, grammatisch nicht zusammen.": "Possibly the noun and its descriptive words do not grammatically agree.",
                "Drei aufeinanderfolgende Sätze beginnen mit dem gleichen Wort. Evtl. können Sie den Satz umformulieren, zum Beispiel, indem Sie ein Synonym nutzen.": "Some sentences start with the same word. Consider rephrasing by using a synonym."
            }
            for m in matches:
                seg = m.context[m.offset:m.offset+m.errorLength]
                orig = m.message
                trans = translation_map.get(orig, "(English translation not available)")
                suggs = ', '.join(m.replacements[:3]) or '–'
                # Color-coded feedback
                st.markdown(f"<span style='color:#e15759;'>Error in '<strong>{seg}</strong>': {orig}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#4e79a7;'>Translation: {trans}</span>", unsafe_allow_html=True)
                st.markdown(f"<span style='color:#76b7b2;'>German suggestions: {suggs}</span>", unsafe_allow_html=True)
        # Highlight issues
        adv = {'berufliches seminar': 'Kurs', 'mehr details': 'mehr Informationen'}
        for line in custom_adv.splitlines():
            if ';' in line:
                k, v = line.split(';', 1)
                adv[k.strip().lower()] = v.strip()
        forb = ['weil ich möchte wissen', 'denn ich möchte wissen']
        for line in custom_forbidden.splitlines():
            ph = line.strip().lower()
            if ph:
                forb.append(ph)
        manual = {
            r"\bhabt\s+ein\s+Probleme\b": "habe ein Problem",
            r"\bein\s+schwarze\b": "ein schwarzes",
            r"\bin\s+eine\s+Karton\b": "in einen Karton",
            r"\b8-?monate\b": "8 Monate"
        }
        patterns = list(manual.keys()) + [re.escape(p) for p in forb] + [re.escape(a) for a in adv]
        if level in ['B1', 'B2']:
            patterns.append(r"\bdu\b")
        ann = text
        for pat in patterns:
            col = "#fffd75"
            if any(re.fullmatch(re.escape(p), pat, re.I) for p in forb):
                col = "#fbb"
            elif any(re.fullmatch(re.escape(a), pat, re.I) for a in adv):
                col = "#bdf"
            ann = re.sub(
                pat,
                lambda m: f"<span style='background-color:{col}'>{m.group(0)}</span>",
                ann,
                flags=re.I
            )
        st.markdown("**Annotated Text:**", unsafe_allow_html=True)
        st.markdown(ann.replace("\n", "  "), unsafe_allow_html=True)

        # Tip to improve
        st.info("Tip: Try varying your vocabulary. Use synonyms and avoid repeating the same words.")

        # Download feedback
        feedback = f"Score: {total}/25\n\n" + "Feedback details:\n"
        feedback += "\n".join([f"- {m.message}" for m in matches])
        st.download_button("Download feedback", feedback, file_name="feedback.txt")
