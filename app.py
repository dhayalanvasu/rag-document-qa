import streamlit as st
import tempfile
from work import ask, chunk_text, mira, index, chunks, load_pdf
import faiss
import numpy as np
import requests 
import os

st.set_page_config(page_title="RAG Q&A", layout="wide")
st.title("RAG Q&A")
st.markdown("Upload a PDF, ask questions and get answers right away!")

#sidebar header
st.sidebar.header("Upload your PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")


if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None   # Will hold the FAISS index

if "chunks" not in st.session_state:
    st.session_state.chunks = []          # Will hold the text chunks

if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# This block runs only when a file has been uploaded.

if uploaded_file is not None:

    # Streamlit gives us a file-like object, but load_pdf() needs a real file PATH.
    # So we save the upload temporarily to disk using tempfile.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())   # Write the uploaded bytes to disk
        tmp_path = tmp.name               # Remember where it was saved

    # Show a spinner while processing — so the user knows something is happening
    with st.spinner("Reading and indexing your PDF..."):

        # Step A: Extract raw text from the PDF
        raw_text = load_pdf(tmp_path)

        # Step B: Split into 100-word chunks
        chunks = chunk_text(raw_text, chunk_size=100)

        # Step C: Embed all chunks using Mira (SentenceTransformer)
        chunk_embeddings = np.array(mira.encode(chunks)).astype('float32')

        # Step D: Build a fresh FAISS index for this PDF
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)

        # Step E: Store everything in session_state so it survives reruns
        st.session_state.faiss_index = index
        st.session_state.chunks = chunks
        st.session_state.pdf_loaded = True

    # Clean up the temp file — we don't need it anymore
    os.unlink(tmp_path)

    # Confirm success to the user
    st.sidebar.success(f"✅ PDF loaded! {len(chunks)} chunks indexed.")

# ── Main area — Q&A section ───────────────────────────────────────────────────

if st.session_state.pdf_loaded:

    st.subheader("Step 2: Ask a question about your document")

    # Text input box — user types their question here
    question = st.text_input(
        label="Your question",
        placeholder="e.g. What is the main finding of the paper?"
    )

    # Ask button — triggers the RAG pipeline
    if st.button("Ask") and question.strip():

        with st.spinner("Thinking..."):

            # We need to pass our session's index and chunks to ask()
            # Because work.py's global index was built on sample.pdf,
            # but we want to use the one built from the uploaded PDF.
            # So we temporarily override them here.

            import work
            work.index = st.session_state.faiss_index
            work.chunks = st.session_state.chunks

            # Call ask() — now returns (answer, retrieved_chunks)
            answer, retrieved_chunks = ask(question)

        # Display the answer prominently
        st.markdown("### 💬 Answer")
        st.write(answer)

        # Show the source chunks in an expandable section
        with st.expander("📚 Source chunks used to answer"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk}")

else:
    # If no PDF uploaded yet, show a friendly prompt
    st.info("👈 Upload a PDF from the sidebar to get started.")

