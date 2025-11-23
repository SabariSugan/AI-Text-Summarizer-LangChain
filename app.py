import streamlit as st
from groq import Groq
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# Use PromptTemplate (STRING-BASED, works on Python 3.13)
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="""
Summarize the following text into 4–6 clear sentences.
Include an introduction, main points, and a conclusion.

Text:
{text}
"""
)

def summarize_text(text):
    # Generate final string prompt
    prompt_str = prompt_template.format(text=text)

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt_str}],
        temperature=0.2,
    )

    return response.choices[0].message["content"]


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Type text to summarize"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(user_input)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)
