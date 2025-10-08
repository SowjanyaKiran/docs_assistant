# app.py
import streamlit as st
import os
import numpy as np
import faiss
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------
# 1. Load PDFs and split into chunks
# ----------------------------------------------------------
st.set_page_config(page_title="📱 Gadget PDF Assistant", layout="wide")

st.title("📘 Gadget PDF Assistant")
st.markdown("Ask questions about your uploaded gadget manuals, brochures, or specs (PDFs).")

# Folder for your PDF files
DATA_FOLDER = "Docs"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# PDF Upload
uploaded_files = st.file_uploader("📂 Upload your Gadget PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        with open(os.path.join(DATA_FOLDER, file.name), "wb") as f:
            f.write(file.getbuffer())
    st.success(f"✅ {len(uploaded_files)} file(s) saved successfully!")

# ----------------------------------------------------------
# 2. Load documents using LangChain loaders
# ----------------------------------------------------------
loader = DirectoryLoader(DATA_FOLDER, glob="**/*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

if not documents:
    st.warning("Please upload PDFs to continue.")
    st.stop()

# ----------------------------------------------------------
# 3. Chunking documents
# ----------------------------------------------------------
st.write("📑 Splitting PDFs into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
st.write(f"Total Chunks: {len(chunks)}")

# ----------------------------------------------------------
# 4. Create SentenceTransformer embeddings
# ----------------------------------------------------------
st.write("🧠 Generating embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # small & fast
texts = [chunk.page_content for chunk in chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# ----------------------------------------------------------
# 5. Build FAISS index
# ----------------------------------------------------------
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
st.write(f"✅ FAISS index built with {index.ntotal} vectors.")

# ----------------------------------------------------------
# 6. Question Answer Retrieval
# ----------------------------------------------------------
st.subheader("💬 Ask a Question About Your Gadgets")

query = st.text_input("Type your question:")

if st.button("🔍 Get Answer"):
    if query.strip() == "":
        st.warning("Please enter a question.")
    else:
        query_embedding = model.encode([query], convert_to_numpy=True)
        D, I = index.search(query_embedding, k=3)

        results = [texts[i] for i in I[0]]

        # Display most relevant context
        st.write("### 📄 Top Matching Contexts:")
        for i, res in enumerate(results):
            with st.expander(f"Context {i+1}"):
                st.write(res)

        # Generate simple answer (local summarization)
        st.write("### 🤖 Assistant’s Answer:")
        answer_text = (
            "Based on the most relevant context:\n\n" + results[0][:800] + "..."
            if results
            else "Sorry, no relevant information found in the PDFs."
        )
        st.info(answer_text)

# ----------------------------------------------------------
# 7. Footer
# ----------------------------------------------------------
st.markdown("---")
st.caption("⚙️ Offline Gadget PDF Assistant | Built with SentenceTransformer + FAISS + Streamlit")
