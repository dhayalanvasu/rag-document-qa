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
import os

# ── STEP 1: Our document chunks ─────────────────────────────────
# Still pretend chunks for now — next session we'll load real PDFs
chunks = [
    "Fishermen should wear life jackets in rough seas.",
    "Maritime safety equipment reduces drowning risk.",
    "Weather prediction models help avoid storms at sea.",
    "The stock market crashed due to inflation fears.",
    "Neural networks learn patterns from large datasets.",
    "IoT sensors can detect water levels and send alerts.",
]

# ── STEP 2: Embed all chunks with Mira ──────────────────────────
mira = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = np.array(mira.encode(chunks)).astype('float32')

# ── STEP 3: Build the FAISS index ───────────────────────────────
dimension = chunk_embeddings.shape[1]   # 384
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print(f"Filing cabinet ready. Chunks stored: {index.ntotal}")

# ── STEP 4: Define the full RAG function ────────────────────────
# This is our complete pipeline in one reusable recipe:
# question → embed → retrieve → prompt → LLM → answer

def ask(question):
    # 4a. Convert the question to a vector
    question_vec = np.array(mira.encode([question])).astype('float32')

    # 4b. Search FAISS for the 3 nearest chunks
    distances, indices = index.search(question_vec, k=3)

    # 4c. Collect the actual text of those chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # 4d. Build the context string
    context = "\n".join([f"- {chunk}" for chunk in retrieved_chunks])

    # 4e. Build the full prompt
    prompt = f"""You are a helpful assistant. Answer the question using ONLY 
the context provided below. If the answer is not in the context, 
say "I don't know based on the provided context."

Context:
{context}

Question: {question}"""

    # 4f. Send to Ollama
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    # 4g. Extract the answer
    answer = response.json()["response"]

    # 4h. Print everything
    print("\n" + "="*50)
    print(f"QUESTION:\n{question}")
    print(f"\nRETRIEVED CHUNKS:")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"  {i+1}. {chunk}")
    print(f"\nANSWER:\n{answer}")
    print("="*50)

# ── STEP 5: Ask two questions ────────────────────────────────────
# One that's answerable from our chunks, one that isn't

ask("What safety gear should sailors use in bad weather?")
ask("Who won the FIFA World Cup in 2022?")