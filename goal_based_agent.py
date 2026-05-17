import streamlit as st
import os
import re
import fitz
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

# -----------------------------
# Application Data Storage
# -----------------------------
application_info = {
    "name": None,
    "email": None,
    "skills": None
}

# -----------------------------
# Extract user input
# -----------------------------
def extract_application_info(text: str) -> str:
    name_match = re.search(
        r"(?:my name(?: is)?|i am|name)\s*[:,-]?\s*([A-Za-z]+(?:\s+[A-Za-z]+)*)",
        text,
        re.IGNORECASE
    )

    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)

    skills_match = re.search(
        r"(?:skills|skills are|i know|i can use)\s*[:,-]?\s*(.+)",
        text,
        re.IGNORECASE
    )

    response = []

    if name_match:
        application_info["name"] = name_match.group(1).title()
        response.append("Name saved.")

    if email_match:
        application_info["email"] = email_match.group(0)
        response.append("Email saved.")

    if skills_match:
        application_info["skills"] = skills_match.group(1).strip()
        response.append("Skills saved.")

    if not any([name_match, email_match, skills_match]):
        return "Please provide your name, email, and skills."

    return " ".join(response)

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""

    for page in doc:
        text += page.get_text()

    return text

# -----------------------------
# Extract CV Info
# -----------------------------
def extract_info_from_cv(text: str):
    extracted_info = {
        "name": None,
        "email": None,
        "skills": None
    }

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if lines:
        if re.match(r"^[A-Za-z]+\s+[A-Za-z]+$", lines[0]):
            extracted_info["name"] = lines[0]

    email_match = re.search(r"\b[\w.-]+@[\w.-]+\.\w+\b", text)

    if email_match:
        extracted_info["email"] = email_match.group(0)

    skills_match = re.search(
        r"(?:Skills|Technical Skills|Skills and Interests)\s*(.*?)(?:\n\n|\Z)",
        text,
        re.IGNORECASE | re.DOTALL
    )

    if skills_match:
        extracted_info["skills"] = skills_match.group(1).replace("\n", ", ").strip()

    return extracted_info

# -----------------------------
# Check Completion
# -----------------------------
def check_application_goal(_: str) -> str:
    if all(application_info.values()):
        return (
            f"You're ready!\n\n"
            f"Name: {application_info['name']}\n"
            f"Email: {application_info['email']}\n"
            f"Skills: {application_info['skills']}"
        )
    else:
        missing = [k for k, v in application_info.items() if not v]
        return f"Still need: {', '.join(missing)}."

# -----------------------------
# LangChain Tools
# -----------------------------
tools = [
    Tool(
        name="extract_application_info",
        func=extract_application_info,
        description="Extract name, email, and skills."
    ),
    Tool(
        name="check_application_goal",
        func=check_application_goal,
        description="Check if all information is collected.",
        return_direct=True
    )
]

# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -----------------------------
# Agent
# -----------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=False
)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="AI Job Application Assistant",
    layout="centered"
)

st.title("AI Job Application Assistant")

st.markdown(
    """
Upload your CV or enter your details manually.

Provide:
- Name
- Email
- Skills
"""
)

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "application_summary" not in st.session_state:
    st.session_state.application_summary = ""

# -----------------------------
# Resume Upload
# -----------------------------
st.sidebar.header("Upload Resume")

resume = st.sidebar.file_uploader(
    "Upload PDF Resume",
    type=["pdf"]
)

if resume:
    text = extract_text_from_pdf(resume)
    extracted = extract_info_from_cv(text)

    for key in application_info:
        if extracted[key]:
            application_info[key] = extracted[key]

    st.sidebar.success("Resume processed!")

    for key, value in extracted.items():
        st.sidebar.write(f"**{key.capitalize()}**: {value}")

# -----------------------------
# Reset
# -----------------------------
if st.sidebar.button("Reset"):
    st.session_state.chat_history = []
    st.session_state.application_summary = ""

    for key in application_info:
        application_info[key] = None

    st.rerun()

# -----------------------------
# Chat Input
# -----------------------------
user_input = st.chat_input("Enter your details...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    extract_application_info(user_input)

    status = check_application_goal("check")

    st.session_state.chat_history.append(("assistant", status))

    if all(application_info.values()):
        st.session_state.application_summary = f"""
Name: {application_info['name']}
Email: {application_info['email']}
Skills: {application_info['skills']}
"""

# -----------------------------
# Chat Display
# -----------------------------
for sender, message in st.session_state.chat_history:
    with st.chat_message(sender):
        st.markdown(message)

# -----------------------------
# Download
# -----------------------------
if st.session_state.application_summary:
    st.success("Application Complete!")

    st.download_button(
        label="Download Summary",
        data=st.session_state.application_summary,
        file_name="application_summary.txt",
        mime="text/plain"
    )