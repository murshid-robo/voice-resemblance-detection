# build_reciter_embeddings.py (fixed)
import os
import glob
import numpy as np
import soundfile as sf
from resemblyzer import VoiceEncoder, preprocess_wav
import warnings

# CONFIG
RECITERS_DIR = "data/reciters"
OUTPUT_FILE = "reciter_embeddings.npz"
MIN_CLIP_DURATION_SEC = 1.5  # skip clips shorter than this
TARGET_SR = 16000            # Resemblyzer uses 16k

encoder = VoiceEncoder()  # loads pretrained model

def load_wav_as_np(path):
    # preprocess_wav will resample to 16k and convert to mono
    wav = preprocess_wav(path)
    return wav

def compute_embedding_for_file(path):
    wav = load_wav_as_np(path)
    duration = len(wav) / TARGET_SR
    if duration < MIN_CLIP_DURATION_SEC:
        raise ValueError(f"Clip too short ({duration:.2f}s) - min {MIN_CLIP_DURATION_SEC}s")
    emb = encoder.embed_utterance(wav)  # 256-d vector
    return emb

def build_reciter_centroids(reciters_dir):
    reciter_centroids = {}
    reciter_counts = {}
    for reciter_folder in sorted(os.listdir(reciters_dir)):
        folder_path = os.path.join(reciters_dir, reciter_folder)
        if not os.path.isdir(folder_path):
            continue
        embeddings = []
        for wav_path in glob.glob(os.path.join(folder_path, "*")):
            # skip hidden files
            if os.path.basename(wav_path).startswith("."):
                continue
            try:
                emb = compute_embedding_for_file(wav_path)
                embeddings.append(emb)
                print(f"OK: {reciter_folder} <- {os.path.basename(wav_path)}")
            except Exception as e:
                print(f"SKIP {wav_path}: {e}")
        if embeddings:
            centroid = np.mean(np.stack(embeddings), axis=0)
            # L2-normalize centroid
            centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
            reciter_centroids[reciter_folder] = centroid
            reciter_counts[reciter_folder] = len(embeddings)
            print(f"Built centroid for {reciter_folder} from {len(embeddings)} clips.")
        else:
            print(f"No valid clips found for {reciter_folder}, skipping.")
    return reciter_centroids, reciter_counts

if __name__ == "__main__":
    if not os.path.isdir(RECITERS_DIR):
        raise SystemExit(f"Reciters directory not found: {RECITERS_DIR}")

    reciter_centroids, reciter_counts = build_reciter_centroids(RECITERS_DIR)

    if len(reciter_centroids) == 0:
        raise SystemExit("No reciter centroids were built. Check your WAV files and their durations.")

    # Save as npz with consistent order
    names = list(reciter_centroids.keys())
    embeddings = np.stack([reciter_centroids[n] for n in names])
    counts = np.array([reciter_counts[n] for n in names])
    np.savez_compressed(OUTPUT_FILE, names=np.array(names, dtype=object), embeddings=embeddings, counts=counts)
    print(f"Saved {len(names)} reciter centroids to {OUTPUT_FILE}")
