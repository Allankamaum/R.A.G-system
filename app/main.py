import os
from dotenv import load_dotenv
import streamlit as st
from rag_pipe import initialize_vectorstore, get_answer


load_dotenv()

st.set_page_config(page_title="📄 RAG Document QA")
st.title("📄 Ask My Document")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")

if uploaded_file:
    with open("app/uploaded.txt", "wb") as f:
        # Make a copy of your file and save it to the app folder
        f.write(uploaded_file.read())
    vectorstore = initialize_vectorstore("app/uploaded.txt")

    st.success("✅ Document uploaded and processed!")
else:
    st.warning("Please upload a .txt file dockto begin.")
    st.stop()

query = st.text_input("Ask a question about the document")
if query:
    answer = get_answer(query, vectorstore)
    st.write("### Answer:")
    st.write(answer)

if "chat_history" in st.session_state:
    del st.session_state["chat_history"]
    st.rerun()