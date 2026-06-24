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
import fitz
import os


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import fitz  # PyMuPDF
import os

# ── STEP 1: Load chunks — from PDF if available, else fallback ──

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

# Check if a real PDF exists — if yes, use it; if no, use fake chunks
PDF_PATH = "sample.pdf"  # put any PDF in your Project folder with this name

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

# ── STEP 2: Embed all chunks ─────────────────────────────────────
mira = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = np.array(mira.encode(chunks)).astype('float32')

# ── STEP 3: Build the FAISS index ────────────────────────────────
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print(f"Filing cabinet ready. Chunks stored: {index.ntotal}")

# ── STEP 4: RAG function ──────────────────────────────────────────
def ask(question):
    # 4a. Embed the question
    question_vec = np.array(mira.encode([question])).astype('float32')

    # 4b. Retrieve top 3 chunks
    distances, indices = index.search(question_vec, k=3)
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # 4c. Build context
    context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

    # 4d. Prompt — loosened so it can infer, not just copy
    prompt = f"""You are a helpful assistant. Answer the question using the context below.
If the answer is directly stated OR can be reasonably inferred from the context, answer it.
Only say "I don't know" if the context has absolutely nothing relevant.

Context:
{context}

Question: {question}"""

    # 4e. Send to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    print("\n" + "="*50)
    print(f"QUESTION:\n{question}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")
    print(f"\nANSWER:\n{answer}")
    print("="*50)

# ── STEP 5: Test it ───────────────────────────────────────────────
ask("When does the KDD exam registration open?")
ask("Who is the lecturer for the guest lecture?")
# ── STEP 1: Our document chunks ─────────────────────────────────
# Still pretend chunks for now — next session we'll load real PDFs
