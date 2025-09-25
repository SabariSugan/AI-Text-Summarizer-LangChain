import streamlit as st
from io import StringIO
import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_community.llms import Ollama

st.set_page_config(page_title="AI Text Summarizer", layout="wide")
st.title("AI Text Summarizer")


text_input = st.text_area("Enter your text here:", height=200)
uploaded_file = st.file_uploader("Or upload a text/PDF file", type=["txt", "pdf"])

content = text_input

if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        content = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()

if st.button("Summarize"):
    if not content.strip():
        st.error("Please enter text or upload a file to summarize.")
    else:
        llm = Ollama(model="llama2")
        splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(content)
        docs = [Document(page_content=chunk) for chunk in chunks]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        summary = chain.run(docs)
        st.subheader("📝 Summary:")
        st.success(summary)
