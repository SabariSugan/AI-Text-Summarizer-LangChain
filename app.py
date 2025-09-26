import streamlit as st
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests

# ✅ Use HF API Key from Streamlit Secrets
HF_API_KEY = st.secrets["HF_API_KEY"]
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"

# Custom Hugging Face LLM
class HF_API_LLM(LLM):
    def _call(self, prompt: str, stop=None) -> str:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        try:
            response = requests.post(API_URL, headers=headers, json={"inputs": prompt}, timeout=60)
            response.raise_for_status()
            return response.json()[0]["summary_text"]
        except Exception as e:
            return f"⚠ Error generating summary: {e}"

    @property
    def _identifying_params(self):
        return {"model": "facebook/bart-large-cnn"}

    @property
    def _llm_type(self):
        return "hf_api"

# Initialize LLM
llm = HF_API_LLM()

# Streamlit app configuration
st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("🤖 AI Summarizer")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to deduplicate repeated sentences
def deduplicate_text(text):
    sentences = text.split(". ")
    seen = set()
    unique_sentences = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean and s_clean not in seen:
            unique_sentences.append(s_clean)
            seen.add(s_clean)
    return ". ".join(unique_sentences)

# Function to summarize text
def summarize_text(text):
    cleaned_text = deduplicate_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(cleaned_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    # Prompt template
    prompt_template = """
You are an expert summarizer. Read the text carefully and summarize it including all important points.
Make sure to include introduction, main points, and conclusion. Write in 4-6 clear sentences.
{text}
"""
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    # Load summarize chain
    chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompt,
        combine_prompt=prompt
    )
    return chain.run(docs)

# Display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if user_input := st.chat_input("Type the text to summarize:"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Summarizing..."):
            summary = summarize_text(user_input)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.write(summary)
