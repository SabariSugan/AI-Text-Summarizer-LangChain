import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests

# Load env
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")

if "messages" not in st.session_state:
    st.session_state.messages = []

def summarize_chunk(prompt_text):
    payload = {"inputs": prompt_text}
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
    try:
        response.raise_for_status()
        return response.json()[0]["summary_text"]
    except Exception as e:
        return f"Error: {e}"

def summarize_text(text):
    # Split long text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    # Prepare prompt
    prompt_template = """
    You are an expert summarizer. Read the text carefully and summarize it including all important points.
    Include introduction, main points, and conclusion. Write in 4-6 clear sentences.
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)
    summarize_chain = load_summarize_chain(
        llm=None,  # We bypass LangChain LLM since we use direct HF API calls
        chain_type="map_reduce",
        return_intermediate_steps=False,
        map_prompt=prompt,
        combine_prompt=prompt
    )

    # Run summaries on each chunk
    summaries = [summarize_chunk(c.page_content) for c in docs]
    combined_summary = " ".join(summaries)
    final_summary = summarize_chunk(combined_summary)
    return final_summary

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if user_input := st.chat_input("Type the Text to Summarize:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(user_input)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)
