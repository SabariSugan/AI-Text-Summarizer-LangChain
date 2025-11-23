import streamlit as st
from groq import Groq
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


prompt_template = ChatPromptTemplate.from_template("""
Summarize the following text into 4–6 clear sentences.
Include an introduction, the main ideas, and a conclusion.

Text:
{text}
""")


def summarize_text(text):
   
    prompt_value = prompt_template.format(text=text)
    

    prompt_str = prompt_value.to_messages()[0]["content"]


    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt_str}],
        temperature=0.2,
    )

    return response.choices[0].message["content"]


-
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
