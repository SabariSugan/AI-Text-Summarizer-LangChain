import streamlit as st
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def groq_llm(prompt: str):
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message["content"]

prompt_template = ChatPromptTemplate.from_template("""
Summarize the following text into 4–6 clear sentences.
Include an introduction, main ideas, and a conclusion.

Text:
{text}
""")

def run_chain(text):
    prompt = prompt_template.format(text=text)
    return groq_llm(prompt)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Type text to summarize"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Summarizing..."):
            summary = run_chain(user_input)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)
