import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import requests
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Custom Hugging Face API LLM with tokenizer
class HF_API_LLM(LLM):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    def _call(self, prompt: str, stop=None) -> str:
        payload = {"inputs": prompt}
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        try:
            response.raise_for_status()
            return response.json()[0]["summary_text"]
        except Exception as e:
            return f"Error: {e}"

    def get_num_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    @property
    def _identifying_params(self):
        return {"model": "facebook/bart-large-cnn"}

    @property
    def _llm_type(self):
        return "hf_api"

# Initialize LLM
llm = HF_API_LLM()

# Streamlit setup
st.set_page_config(page_title="AI Summarizer", layout="wide")
st.title("AI Summarizer")

if "messages" not in st.session_state:
    st.session_state.messages = []

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

def summarize_text(text):
    cleaned_text = deduplicate_text(text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(cleaned_text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    prompt_template = """
    You are an expert summarizer. Read the text carefully and summarize it including all important points. 
    Make sure to include introduction, main points, and conclusion. Write in 4-6 clear sentences.
    {text}
    """
    prompt = PromptTemplate(input_variables=["text"], template=prompt_template)

    summarize_chain = load_summarize_chain(
        llm,
        chain_type="map_reduce",
        return_intermediate_steps=False,
        map_prompt=prompt,
        combine_prompt=prompt
    )

    final_summary = summarize_chain.run(docs)
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
