import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")


llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.3-70b-versatile",
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_template("""
Summarize the following text into 4–6 clear sentences.
Include an introduction, key ideas, and a conclusion.

Text:
{text}
""")

chain = prompt | llm | StrOutputParser()

def summarize_text(text):
    return chain.invoke({"text": text})

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input("Type text to summarize"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Summarizing..."):
            try:
                summary = summarize_text(user_input)
            except Exception as e:
                summary = f"Error summarizing text: {e}"
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)

