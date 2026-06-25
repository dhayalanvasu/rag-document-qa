# Three tools today:
# sentence_transformers — Mira, converts text to vectors
# faiss               — the fast filing cabinet
# numpy               — number calculator
# anthropic           — the detective (Claude LLM)
# os                  — lets Python read environment variables
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF
import os

# ── STEP 1: Functions ─────────────────────────────────────────────

def load_pdf(filepath):
    """Open a PDF and extract all text, page by page."""
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=100):
    """Split a long text into smaller word chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# ── Always-available globals (app.py imports these) ──────────────
mira = SentenceTransformer('all-MiniLM-L6-v2')
index = None
chunks = []

# ── STEP 4: RAG function ──────────────────────────────────────────
def ask(question):
    question_vec = np.array(mira.encode([question])).astype('float32')
    distances, indices = index.search(question_vec, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

    prompt = f"""You are a helpful assistant. Answer the question using the context below.
If the answer is directly stated OR can be reasonably inferred from the context, answer it.
Only say "I don't know" if the context has absolutely nothing relevant.

Context:
{context}

Question: {question}"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.2", "prompt": prompt, "stream": False}
    )

    answer = response.json()["response"]

    print("\n" + "="*50)
    print(f"QUESTION:\n{question}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")
    print(f"\nANSWER:\n{answer}")
    print("="*50)
    return answer, retrieved_chunks

# ── Only runs when: python3 work.py directly ─────────────────────
if __name__ == "__main__":
    PDF_PATH = "sample.pdf"

    if os.path.exists(PDF_PATH):
        print(f"PDF found! Loading: {PDF_PATH}")
        raw_text = load_pdf(PDF_PATH)
        chunks = chunk_text(raw_text, chunk_size=100)
        print(f"Total chunks created from PDF: {len(chunks)}")
    else:
        print("No PDF found. Using test chunks.")
        chunks = [
            "Fishermen should wear life jackets in rough seas.",
            "Maritime safety equipment reduces drowning risk.",
            "Weather prediction models help avoid storms at sea.",
            "The stock market crashed due to inflation fears.",
            "Neural networks learn patterns from large datasets.",
            "IoT sensors can detect water levels and send alerts.",
        ]

    chunk_embeddings = np.array(mira.encode(chunks)).astype('float32')
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)
    print(f"Filing cabinet ready. Chunks stored: {index.ntotal}")

    ask("When does the KDD exam registration open?")
    ask("Who is the lecturer for the guest lecture?")