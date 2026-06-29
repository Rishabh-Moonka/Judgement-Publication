import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import os

# =====================
# PATH SETUP
# =====================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# =====================
# CONFIG
# =====================
TRAIN_CSV = os.path.join(PROJECT_DIR, "data/train_offense_facts.csv")
COLLECTION_NAME = "offense_facts"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 5000
CHROMA_DB_DIR = os.path.join(PROJECT_DIR, "chroma_rag_db")

print(f"🔧 Chroma persist dir: {CHROMA_DB_DIR}")

# =====================
# LOAD DATA
# =====================
df = pd.read_csv(TRAIN_CSV)

# Required columns
required_cols = ["only_facts", "label"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

assert df["only_facts"].notna().all(), "❌ NaNs found in only_facts"
assert df["label"].isin([0, 1]).all(), "❌ Invalid label values found"

docs = df["only_facts"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

print(f"✅ Loaded {len(docs)} documents")

# =====================
# INIT CHROMA (PERSISTENT)
# =====================
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

# =====================
# LOAD EMBEDDER
# =====================
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
embedder = SentenceTransformer(EMBED_MODEL, device=device)

print(f"✅ Embedder running on: {embedder.device}")

# =====================
# BUILD DB (BATCHED)
# =====================
for start in tqdm(range(0, len(docs), BATCH_SIZE)):
    end = min(start + BATCH_SIZE, len(docs))

    batch_docs = docs[start:end]
    batch_labels = labels[start:end]

    batch_ids = [str(i) for i in range(start, end)]

    # Metadata: store bail outcome ONLY as metadata
    batch_metadata = [
        {
            "label": int(lbl),  # 1 = bail granted, 0 = bail denied
            "label_text": "Bail Granted" if lbl == 1 else "Bail Denied"
        }
        for lbl in batch_labels
    ]

    batch_embeddings = embedder.encode(
        batch_docs,
        batch_size=64,
        show_progress_bar=False
    ).tolist()

    collection.add(
        documents=batch_docs,
        embeddings=batch_embeddings,
        metadatas=batch_metadata,
        ids=batch_ids
    )

print("🎉 ChromaDB successfully built and saved!")
print(f"📂 Location: {CHROMA_DB_DIR}")
print(f"📦 Collection name: {COLLECTION_NAME}")