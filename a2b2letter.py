import streamlit as st
import re
import requests
from fpdf import FPDF

st.set_page_config(page_title="German Letter & Essay Checker - Learn Language Education Academy")

st.title("📄 German Letter & Essay Checker")
st.subheader("By Learn Language Education Academy")

st.markdown("---")
st.markdown("### ℹ Important Note for Students")
st.markdown("""
✅ **Greetings** like *Hallo Felix*, *Liebe Anna*, etc. are checked automatically.

✅ **Reason for writing** like *Ich schreibe Ihnen...* or *Ich schreibe dir...* is checked automatically.

❌ **Closing remarks** (*Mit freundlichen Grüßen*, *Viele Grüße*, etc.) are **not scored or checked by the app**.

✅ **Spelling mistakes in names** (like *Porsh*) might be flagged. You can ignore these if it’s your correct name.
""")
st.markdown("---")

level = st.selectbox("Step 1: Select your level", ["A2", "B1", "B2"])
task_type = st.selectbox(
    "Step 2: Select your task type",
    ["Formal Letter", "Informal Letter", "Opinion Essay"]
)

letter_question = st.text_area("Step 3: Enter your question or topic:", height=150)
student_letter = st.text_area("Step 4: Write your letter or essay below:", height=400)

submit = st.button("✅ Submit for Feedback")

# ======= Helper Functions =========

def check_greeting(text, task_type):
    text_normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if task_type == "Formal Letter":
        greetings = ["sehr geehrte", "sehr geehrter", "sehr geehrte damen und herren"]
    elif task_type == "Informal Letter":
        greetings = ["hallo", "liebe", "lieber"]
    else:
        return True
    return any(greet in text_normalized for greet in greetings)

def check_reason(text, task_type):
    text_normalized = re.sub(r"\s+", " ", text.lower()).strip()
    reason_phrases = [
        "ich schreibe",
        "ich schreibe dir",
        "ich schreibe ihnen",
        "ich schreibe ihnen weil",
        "ich schreibe dir weil"
    ]
    if task_type == "Opinion Essay":
        return True
    return any(phrase in text_normalized for phrase in reason_phrases)

def highlight_errors(text):
    errors = []
    if "mochte" in text.lower() and "möchte" not in text.lower():
        errors.append("⚠ **Tip:** Spelling issue. 'mochte' should be 'möchte' (missing umlaut). Suggestion: Replace with 'möchte'.")
    return errors

def spelling_check(text):
    api_url = "https://api.languagetoolplus.com/v2/check"
    params = {"text": text, "language": "de-DE"}
    response = requests.post(api_url, data=params)
    result = response.json()
    mistakes = []
    for match in result.get("matches", []):
        message = match['message'].lower()
        problem_text = match['context']['text']
        if any(skip in message for skip in ["leerzeichen", "spaces", "grußformel", "schlussformel", "closing"]):
            continue
        if "tippfehler" in message or "spelling" in message:
            tip = "⚠ Tip: Possible spelling mistake. Suggestion: Check the word using Google Translate."
        elif "wortstellung" in message:
            tip = "⚠ Tip: Check sentence word order. Suggestion: Review verb position."
        elif "komma" in message:
            tip = "⚠ Tip: Check comma usage. Suggestion: Review punctuation rules."
        elif "übereinstimmung" in message:
            tip = "⚠ Tip: Check verb-subject agreement. Suggestion: Ensure the verb matches the subject."
        elif "modalverb" in message:
            tip = "⚠ Tip: Check modal verb placement. Suggestion: Put modal verbs before the infinitive."
        else:
            tip = "⚠ Tip: Possible spelling, word order, or grammar issue. Suggestion: Read the sentence aloud slowly and check verb position and spelling."
        feedback = f"{tip} | **Problem:** '{problem_text}'"
        if feedback not in mistakes:
            mistakes.append(feedback)
    return mistakes

if submit:
    feedback = []
    score = 25

    spelling_issues = 0
    grammar_issues = 0
    content_issues = 0
    connector_issues = 0
    advanced_grammar_warnings = 0

    # --- Greeting check ---
    if not check_greeting(student_letter, task_type):
        feedback.append("❌ **Proper greeting is missing or incorrect.** Suggestion: Use 'Sehr geehrte Damen und Herren' for formal letters or 'Hallo Max' for informal letters.")
        score -= 3
        grammar_issues += 1

    # --- Reason check ---
    if not check_reason(student_letter, task_type):
        feedback.append("❌ **Reason for writing is missing (use 'Ich schreibe...').** Suggestion: Start with 'Ich schreibe Ihnen, weil...' or 'Ich schreibe dir, weil...'.")
        score -= 3
        grammar_issues += 1

    # --- Highlight common errors ---
    structure_errors = highlight_errors(student_letter)
    if structure_errors:
        feedback.extend(structure_errors)
        score -= len(structure_errors)
        spelling_issues += len(structure_errors)

    # --- Spelling and grammar check ---
    spelling_errors = spelling_check(student_letter)
    if spelling_errors:
        feedback.extend(spelling_errors)
        score -= min(len(spelling_errors), 5)
        spelling_issues += len(spelling_errors)

    # --- Content relevance ---
    keywords = [w for w in re.findall(r"\w+", letter_question.lower()) if len(w) > 3]
    content_mismatch = sum(1 for w in keywords if w not in student_letter.lower())
    if content_mismatch > 7 and task_type != "Opinion Essay":
        feedback.append("⚠ **Tip:** Your letter content might not match the question/task well. Suggestion: Use important words or ideas from the question in your answer.")
        score -= 2
        content_issues += 1

    # --- Connector usage ---
    connectors = ["weil", "deshalb", "denn", "ich möchte wissen, ob", "wenn", "trotzdem", "außerdem", "damit"]
    connector_usage = sum(len(re.findall(r'\b' + re.escape(c) + r'\b', student_letter.lower())) for c in connectors)
    if connector_usage < 2:
        feedback.append("⚠ **Tip:** Too few connectors. Suggestion: Use 'weil', 'deshalb', 'damit', 'außerdem' to connect ideas.")
        score -= 2
        connector_issues += 1

    # --- Advanced grammar ---
    if level == "A2" and any(w in student_letter for w in ["dessen", "deren", "genitiv", "während"]):
        feedback.append("⚠ **Tip:** This grammar might be too difficult for your level. Suggestion: Use simpler sentences that match A2/B1 level.")
        score -= 1
        advanced_grammar_warnings += 1

    if level in ["A2", "B1"] and re.search(r"\sdes\s|\sderen\s", student_letter):
        feedback.append("⚠ **Tip:** Avoid advanced sentence structures. Suggestion: Write shorter, clearer sentences.")
        score -= 1
        advanced_grammar_warnings += 1

    # --- Word count ---
    word_count = len(student_letter.split())
    if word_count < 40:
        feedback.append("⚠ **Tip:** Your text is too short. Suggestion: Aim for at least 50–80 words by adding more details or examples.")
        score -= 1

    # --- Umlaut tip ---
    if any(v in student_letter and u not in student_letter for v, u in [('o', 'ö'), ('a', 'ä'), ('u', 'ü')]):
        feedback.append("⚠ **Tip:** You might have forgotten umlauts like 'ö', 'ä', or 'ü'. Suggestion: Hold the letter key on your keyboard to see umlaut options.")

    score = max(score, 5)

    # --- Emoji score comment ---
    if score >= 22:
        emoji_comment = "🟢 Excellent! Keep it up."
    elif score >= 15:
        emoji_comment = "🟡 Good progress. Some areas need work."
    else:
        emoji_comment = "🔴 Needs more practice. Review the feedback carefully."

    # ================= PASS/FAIL MESSAGE =================

    if (level == "A2" and score >= 18) or (level in ["B1", "B2"] and score >= 20):
        st.success("🎉 Good news! Your score is strong enough. You can now submit this letter or essay to your tutor for final review.")
    else:
        st.warning("Your writing is developing well but needs some more improvement. Please review the feedback and try to revise before submitting to your tutor. If you feel stuck, you can also ask your tutor for ideas")

    # ================= SHOW SCORE FIRST =================

    st.markdown(f"### 🎯 Your Score: **{score} / 25** {emoji_comment}")

    # ================= SHOW FEEDBACK NEXT =================

    st.markdown("### 📝 Analyse & Feedback")

    if feedback:
        st.markdown("**Here are the suggestions and tips for your writing:**")
        for item in feedback:
            st.markdown(f"- {item}")
    else:
        st.success("✅ Great job! No major issues found.")

    # ================= MISTAKE SUMMARY =================

    st.markdown("---")
    st.markdown("### 🗂 Mistake Summary")
    st.markdown(f"- **Spelling issues:** {spelling_issues}")
    st.markdown(f"- **Grammar issues:** {grammar_issues}")
    st.markdown(f"- **Content relevance issues:** {content_issues}")
    st.markdown(f"- **Connector usage issues:** {connector_issues}")
    st.markdown(f"- **Advanced grammar warnings:** {advanced_grammar_warnings}")


        # ================= PDF GENERATOR FUNCTION =================

    def clean_text(text):
        # Remove/replace problematic characters for PDF
        return (
            text
            .replace("⚠", "- Tip:")
            .replace("❌", "- Tip:")
            .replace("🟢", "")
            .replace("🟡", "")
            .replace("🔴", "")
            .replace("–", "-")
            .replace("—", "-")
            .replace("„", '"')
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("ö", "oe")
            .replace("ä", "ae")
            .replace("ü", "ue")
            .replace("ß", "ss")
            .encode("ascii", "ignore")  # Remove any remaining weird symbols
            .decode()
        )

    def generate_pdf(level, task_type, word_count, score, plain_comment,
                     spelling_issues, grammar_issues, content_issues,
                     connector_issues, advanced_grammar_warnings,
                     feedback, improvement_tips):

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        pdf.cell(0, 10, "German Letter & Essay Feedback Report", ln=True, align="C")
        pdf.cell(0, 10, "Learn Language Education Academy", ln=True, align="C")
        pdf.ln(10)

        pdf.cell(0, 10, f"Level: {level}", ln=True)
        pdf.cell(0, 10, f"Task Type: {task_type}", ln=True)
        pdf.cell(0, 10, f"Word Count: {word_count}", ln=True)
        pdf.cell(0, 10, f"Score: {score} / 25  {plain_comment}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Mistake Summary:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"- Spelling issues: {spelling_issues}", ln=True)
        pdf.cell(0, 10, f"- Grammar issues: {grammar_issues}", ln=True)
        pdf.cell(0, 10, f"- Content relevance issues: {content_issues}", ln=True)
        pdf.cell(0, 10, f"- Connector usage issues: {connector_issues}", ln=True)
        pdf.cell(0, 10, f"- Advanced grammar warnings: {advanced_grammar_warnings}", ln=True)
        pdf.ln(10)

        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detailed Feedback and Suggestions:", ln=True)
        pdf.set_font("Arial", size=12)
        for f in feedback:
            safe_feedback = clean_text(f)
            pdf.multi_cell(0, 10, safe_feedback)

        pdf.ln(10)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "General Improvement Tips:", ln=True)
        pdf.set_font("Arial", size=12)
        for tip in improvement_tips:
            safe_tip = clean_text(tip)
            pdf.multi_cell(0, 10, f"- {safe_tip}")

        pdf.ln(10)
        pdf.multi_cell(0, 10, "Note: If you don't understand any feedback point, please ask your teacher.")
        pdf.multi_cell(0, 10, "Learn Language Education Academy - Empowering your German learning journey.")

        return pdf.output(dest="S").encode("latin1")

    # --- Prepare plain comment without emoji ---
    plain_comment = emoji_comment.replace("🟢", "").replace("🟡", "").replace("🔴", "").strip()

    # --- Closing phrase suggestion ---
    closing_phrase = ""
    if task_type == "Formal Letter":
        if level == "B2":
            closing_phrase = ("Suggested closing phrase: Vielen Dank im Voraus für Ihre Berücksichtigung. "
                              "Falls Sie weitere Informationen benötigen, stehe ich Ihnen jederzeit zur Verfügung.")
        else:
            closing_phrase = "Suggested closing phrase: Vielen Dank im Voraus. Ich freue mich auf Ihre Antwort."
    elif task_type == "Informal Letter":
        closing_phrase = "Suggested closing phrase: Ich freue mich auf deine Antwort. Viele Grüße."

    # --- Improvement tips ---
    if task_type in ["Formal Letter", "Informal Letter"]:
        improvement_tips = [
            "Proper greeting: Use 'Sehr geehrte Damen und Herren' for formal letters or 'Hallo Max' for informal letters.",
            "Reason for writing: Start with 'Ich schreibe Ihnen, weil...' or 'Ich schreibe dir, weil...'.",
            "Spelling mistakes: Check the word using Google Translate and compare with correct German spelling.",
            "Grammar issues: Review sentence word order. Ask your tutor if unsure.",
            "Content relevance issues: Use important words or ideas from the question in your answer.",
            "Connector usage issues: Use 'weil', 'deshalb', 'damit', 'außerdem' to connect ideas.",
            "Advanced grammar: Use simpler sentences that match A2/B1 level.",
            "Text too short: Add examples or reasons to reach 50–80 words.",
            "Umlaut issues: Hold down the letter key (like 'o') to access 'ö', 'ä', 'ü'."
        ]
        if closing_phrase:
            improvement_tips.append(closing_phrase)

    else:  # Opinion Essay
        improvement_tips = [
            "Introduction: Start with a simple sentence that introduces the topic.",
            "Example: Heutzutage ist das Thema Lernen (Summarize the topic here) ein wichtiges Thema in unserem Leben.",
            "Explanation: State the topic and why it is important.",
            "State your opinion: Use 'Ich bin der Meinung, dass...' and explain using 'weil...'.",
            "Advantages: Use 'Einerseits gibt es viele Vorteile...' and add examples starting with 'Zum Beispiel...'.",
            "Disadvantages: Use 'Andererseits gibt es auch Nachteile...' and give an example starting with 'Ein Beispiel dafür ist...'.",
            "Final opinion: Use 'Ich glaube, dass...' to reaffirm your opinion.",
            "Conclusion: Use 'Zusammenfassend lässt sich sagen, dass...' to summarize your key message.",
            "Spelling and grammar: Check spelling and review sentence word order.",
            "Connector usage: Use 'weil', 'deshalb', 'außerdem', 'denn', and 'trotzdem' to connect ideas.",
            "Text length: Aim for at least 80–100 words for essays.",
            "Umlaut issues: Hold down the letter key to access 'ö', 'ä', 'ü'."
        ]

    # ================= SHOW IMPROVEMENT TIPS IN APP =================

    st.markdown("---")
    st.markdown("### 🛠 How to improve your mistakes:")

    for tip in improvement_tips:
        st.markdown(f"- {tip}")

    # ================= PDF DOWNLOAD =================

    pdf_data = generate_pdf(
        level, task_type, word_count, score, plain_comment,
        spelling_issues, grammar_issues, content_issues,
        connector_issues, advanced_grammar_warnings,
        feedback, improvement_tips
    )

    st.markdown("### 📥 Download Your Report")

    downloaded = st.download_button(
        label="📄 Download PDF Report",
        data=pdf_data,
        file_name="German_Writing_Feedback_Report.pdf",
        mime="application/pdf"
    )

    if downloaded:
        st.success("✅ Your PDF report has been prepared. Please check your downloads.")
