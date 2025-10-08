# app.py
import streamlit as st
import os
import numpy as np
import faiss
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------------
# 1. Streamlit setup
# ----------------------------------------------------------
st.set_page_config(page_title="📱 Gadget PDF Assistant", layout="wide")

st.title("📘 Gadget PDF Assistant")
st.markdown("Ask questions about your uploaded gadget manuals, brochures, or specs (PDFs).")

DATA_FOLDER = "Docs"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# ----------------------------------------------------------
# 2. File Upload Section
# ----------------------------------------------------------
uploaded_files = st.file_uploader(
    "📂 Upload your Gadget PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ {len(uploaded_files)} file(s) saved successfully!")

# ----------------------------------------------------------
# 3. Function to build or load FAISS index
# ----------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_index(data_folder):
    loader = DirectoryLoader(data_folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    if not documents:
        return None, None, None

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]

    # Generate embeddings
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return model, index, texts

# ----------------------------------------------------------
# 4. Initialize or reuse session state
# ----------------------------------------------------------
if "index" not in st.session_state:
    with st.spinner("🔍 Loading and indexing your PDFs..."):
        model, index, texts = load_index(DATA_FOLDER)
        if index is None:
            st.warning("Please upload PDFs to continue.")
            st.stop()
        st.session_state.model = model
        st.session_state.index = index
        st.session_state.texts = texts
else:
    model = st.session_state.model
    index = st.session_state.index
    texts = st.session_state.texts

st.success(f"✅ FAISS index loaded with {index.ntotal} chunks.")

# ----------------------------------------------------------
# 5. Question Answering Section
# ----------------------------------------------------------
st.subheader("💬 Ask a Question About Your Gadgets")

query = st.text_input("Type your question here:")

if st.button("🔍 Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a valid question.")
    else:
        with st.spinner("🔎 Searching best matches..."):
            query_embedding = model.encode([query], convert_to_numpy=True)
            D, I = index.search(query_embedding, k=3)

            results = [texts[i] for i in I[0]]

        st.write("### 📄 Top Matching Contexts:")
        for i, res in enumerate(results):
            with st.expander(f"Context {i+1}"):
                st.write(res)

        # Combine and trim context for a simple local answer
        combined_context = " ".join(results)
        answer_text = (
            f"**Answer based on your query '{query}':**\n\n"
            + combined_context[:800]
            + "..."
            if results
            else "Sorry, I couldn't find relevant info in the PDFs."
        )

        st.write("### 🤖 Assistant’s Answer:")
        st.info(answer_text)

# ----------------------------------------------------------
# 6. Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("⚙️ Offline Gadget PDF Assistant | Built with SentenceTransformer + FAISS + Streamlit")
