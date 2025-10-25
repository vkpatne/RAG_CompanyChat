
import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv
load_dotenv()

# Model IDs (public)
DEFAULT_EMBED_MODEL = os.getenv("EMBED_MODEL_ID", "intfloat/e5-base-v2")

# Local directories where models will be stored/loaded from
EMBED_LOCAL_DIR = os.getenv("EMBED_LOCAL_DIR", "C://models//e5_base_v2//")
LLM_LOCAL = os.getenv("LLM_LOCAL","C://models//mistral-7b-instruct-v0.1K_M.gguf")

# Index and docs
INDEX_PATH = os.getenv("INDEX_PATH", "./db/faiss_index.bin")
DOCS_PATH = os.getenv("DOCS_PATH", "./db/docs.pkl")

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))

# LangChain / runtime flags
FORCE_OFFLINE = os.getenv("FORCE_OFFLINE", "false").lower() == "true"

def ensure_local_model(repo_id, local_dir):
    # If already present, use it. Otherwise download once.
    if os.path.exists(os.path.join(local_dir, "config.json")) or os.path.exists(os.path.join(local_dir, "tokenizer.json")):
        return local_dir
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading {repo_id} to {local_dir} ...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    return local_dir

# Resolve model paths (download if missing and not forced offline)
if FORCE_OFFLINE:
    EMBED_MODEL_PATH = EMBED_LOCAL_DIR
    LLM_LOCAL_PATH = LLM_LOCAL
else:
    EMBED_MODEL_PATH = ensure_local_model(DEFAULT_EMBED_MODEL, EMBED_LOCAL_DIR)
    # For LLM, user may have already installed locally; attempt to download if missing
    LLM_LOCAL_PATH = LLM_LOCAL

# When forcing offline, set HF envs
if FORCE_OFFLINE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
