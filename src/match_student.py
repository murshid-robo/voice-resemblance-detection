# match_student.py (fixed)
import sys
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from sklearn.metrics.pairwise import cosine_similarity
import os

RECITER_EMBED_FILE = "reciter_embeddings.npz"
MIN_STUDENT_DURATION_SEC = 1.5
TARGET_SR = 16000

encoder = VoiceEncoder()

def load_reciter_centroids(npzfile):
    if not os.path.exists(npzfile):
        raise FileNotFoundError(f"Reciter embeddings file not found: {npzfile}")
    data = np.load(npzfile, allow_pickle=True)
    names = list(data["names"])
    embeddings = data["embeddings"]
    # ensure embeddings normalized
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    return names, embeddings

def compute_student_embedding(path):
    wav = preprocess_wav(path)
    duration = len(wav) / TARGET_SR
    if duration < MIN_STUDENT_DURATION_SEC:
        raise ValueError(f"Student clip too short ({duration:.2f}s) - min {MIN_STUDENT_DURATION_SEC}s")
    emb = encoder.embed_utterance(wav)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    return emb

def match_student(student_path, top_k=3):
    names, reciter_embs = load_reciter_centroids(RECITER_EMBED_FILE)
    student_emb = compute_student_embedding(student_path)
    sims = cosine_similarity(student_emb.reshape(1, -1), reciter_embs).flatten()
    pairs = list(zip(names, sims.tolist()))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:top_k]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 match_student.py path/to/student.wav")
        sys.exit(1)
    student_wav = sys.argv[1]
    try:
        top_matches = match_student(student_wav, top_k=5)
        print("Top matches:")
        for name, score in top_matches:
            print(f"  {name}: {score:.4f}")
    except Exception as e:
        print("Error:", e)
        sys.exit(1)
