# streamlit_complianceadvisory.py

import streamlit as st
from complianceadvisor import (
    nist_only_chat,
    company_compliance_chat,
    build_company_index,
    policy_store,  # initial default index
)
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(
    page_title="Compliance Advisor Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

mode = st.sidebar.radio(
    "Select mode:",
    ["NIST Lookup", "Policy Gap Analysis"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload your company policy**")

# If a new policy is uploaded, rebuild the index
uploaded = st.sidebar.file_uploader(
    label="Upload your company policy (PDF or TXT)",
    type=["pdf", "txt"]
)

if uploaded:
    # 1) save with correct extension
    ext = Path(uploaded.name).suffix  # e.g. ".pdf" or ".txt"
    policy_path = f"uploaded_policy{ext}"
    with open(policy_path, "wb") as f:
        f.write(uploaded.read())

    # 2) choose loader based on extension
    st.sidebar.info("Indexing uploaded policy…")
    try:
        if ext.lower() == ".pdf":
            docs = PyPDFLoader(policy_path).load_and_split()
        else:
            docs = TextLoader(policy_path).load()  # falls back to TextLoader for .txt

        if not docs:
            raise ValueError("No pages or text found in uploaded file.")

        # 3) chunk + index
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        policy_store = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory="./policy_index")
        policy_retriever = policy_store.as_retriever(search_kwargs={"k": 3})

        st.sidebar.success("Policy indexed! Now ask your question below.")
    except Exception as e:
        st.sidebar.error(f"Failed to index policy: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat"):
    st.session_state.history = []

# ─── MAIN LAYOUT ────────────────────────────────────────────────────────────
st.title("🤖 Compliance Advisor Chatbot")

# initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# two‐column layout: chat on left, instructions on right
chat_col, info_col = st.columns([3, 1])

with chat_col:
    # render past messages
    for role, msg in st.session_state.history:
        st.chat_message(role).write(msg)

    # new user input
    if user_input := st.chat_input("Ask your question…"):
        st.session_state.history.append(("user", user_input))
        st.chat_message("user").write(user_input)

        with st.spinner("Thinking…"):
            if mode == "NIST Lookup":
                reply = nist_only_chat(user_input)
            else:
                reply = company_compliance_chat(user_input)
        st.session_state.history.append(("assistant", reply))
        st.chat_message("assistant").write(reply)

with info_col:
    st.markdown("### How to use")
    st.markdown(
        """
- **NIST Lookup**: Ask any question about NIST SP 800-53 controls.
- **Policy Gap Analysis**: Upload your **own** policy (PDF/TXT) in the sidebar, then ask where it might be missing controls.
- Use **Clear Chat** to reset the conversation.
"""
    )
    st.markdown("### Tips")
    st.markdown(
        """
- For best results, keep questions specific (e.g. “gaps in encryption-at-rest”).
- After uploading a new policy, wait for “Policy indexed!” before asking.
"""
    )
