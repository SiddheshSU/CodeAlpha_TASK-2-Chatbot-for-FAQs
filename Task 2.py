import streamlit as st
import nltk
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Student Support Chatbot",
    page_icon="🎓",
    layout="centered"
)


# ==========================================
# FAQ DATASET
# ==========================================

faq_questions = [
"1. What are the college working hours?",

"2. How can I apply for admission?",

"3. What documents are required for admission?",

"4. How can I pay my college fees?",

"5. How can I contact the college?",

"6. Where is the admission office?",

"7. How can I get my student ID card?",

"8. How can I check my exam timetable?",

"9. When are the exams conducted?",

"10. How can I get my result?"


    
]

faq_answers = [
    "1.The college working hours are from 9 AM to 5 PM.",
    "2.You can apply for admission through the college admission portal.",
    "3.You need academic certificates, identity proof, photographs, and other required documents.",
    "4.You can pay your college fees through the online payment gateway.",
    "5.You can contact the college through the official website or by calling the helpline.",
    "6.The admission office is located on the ground floor of the main building.",
    "7.You can get your student ID card from the student services office after submitting the required documents.",
    "8.You can check your exam timetable on the college website or by contacting the examination department.",
    "9.The exams are conducted twice a year, usually in June and December.",
    "10.You can get your result by logging into the college portal or by contacting the examination department."
]


# ==========================================
# NLP PREPROCESSING
# ==========================================

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        "",
        text
    )

    tokens = nltk.word_tokenize(text)

    return " ".join(tokens)


processed_questions = [
    preprocess_text(question)
    for question in faq_questions
]


# ==========================================
# TF-IDF MODEL
# ==========================================

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(
    processed_questions
)


# ==========================================
# FIND BEST ANSWER
# ==========================================

def get_answer(user_question):

    processed_question = preprocess_text(
        user_question
    )

    user_vector = vectorizer.transform(
        [processed_question]
    )

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match_index = similarities.argmax()

    best_score = similarities[0][best_match_index]

    # Minimum similarity requirement
    if best_score < 0.20:

        return (
            "I'm sorry, I couldn't find a suitable answer "
            "to your question. Please try asking it differently."
        ), best_score

    answer = faq_answers[best_match_index]

    return answer, best_score


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🎓 Student Support")

    st.write(
        "Welcome! I can help you with common "
        "college-related questions."
    )

    st.divider()

    st.subheader("📚 Topics")

    st.write("• Admission")
    st.write("• Fees")
    st.write("• Examinations")
    st.write("• Student ID")
    st.write("• Results")

    st.divider()

    st.caption(
        "Powered by NLP + TF-IDF + Cosine Similarity"
    )


# ==========================================
# MAIN HEADER
# ==========================================

st.title("🎓 Student Support Chatbot")

st.write(
    "Ask a question about college admission, "
    "fees, examinations, student services, or results."
)

st.divider()


# ==========================================
# CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your Student Support "
                "Chatbot. How can I help you today?"
            )
        }
    ]


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ==========================================
# USER INPUT
# ==========================================

user_question = st.chat_input(
    "💬 Type your question here..."
)


if user_question:

    # Add user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    # Display user message
    with st.chat_message("user"):

        st.write(user_question)


    # Find answer
    answer, score = get_answer(
        user_question
    )


    # Add assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    # Display assistant response
    with st.chat_message("assistant"):

        st.write(answer)


# ==========================================
# SUGGESTED QUESTIONS
# ==========================================

st.divider()

st.subheader("💡 Try asking")

col1, col2 = st.columns(2)

with col1:

    st.write("📌 How can I apply for admission?")
    st.write("📌 How can I pay my fees?")
    st.write("📌 How can I get my student ID?")


with col2:

    st.write("📌 When are the exams?")
    st.write("📌 How can I check my result?")
    st.write("📌 What documents are required?")


# ==========================================
# CLEAR CHAT
# ==========================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm your Student Support "
                "Chatbot. How can I help you today?"
            )
        }
    ]

    st.rerun()