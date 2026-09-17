import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
BOOKS_DIR = DATA_DIR / "books"
FRONTEND_DIR = BASE_DIR / "frontend"

for directory in [DATA_DIR, UPLOADS_DIR, BOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Hardware & CPU Parallel Settings
# Explicitly configured for 4 parallel CPU cores as requested
NUM_WORKERS = 4
ONNX_THREADS = 4

# FastEmbed ONNX Model Settings (CPU-optimized)
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# RAG & Retrieval Settings
CHUNK_TARGET_SIZE = 1200      # characters (~250-300 words)
CHUNK_OVERLAP = 200           # characters overlap
TOP_K_RETRIEVAL = 5           # number of hybrid passages to return
RERANK_TOP_K = 4

# LLM Providers Configuration
DEFAULT_LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# =====================================================
# Custom From-Scratch LLM Configuration
# =====================================================
LLM_TRAINING_DIR = DATA_DIR / "llm_training_pdfs"
LLM_TESTING_DIR = DATA_DIR / "llm_testing_pdfs"
LLM_MODELS_DIR = DATA_DIR / "llm_models"

for llm_dir in [LLM_TRAINING_DIR, LLM_TESTING_DIR, LLM_MODELS_DIR]:
    llm_dir.mkdir(parents=True, exist_ok=True)

# Transformer Hyperparameters (GPT-2 style, ~10M params, CPU-trainable)
LLM_VOCAB_SIZE = 4096          # BPE vocabulary size
LLM_CONTEXT_LENGTH = 256       # max sequence length (tokens)
LLM_EMBED_DIM = 384            # embedding dimension
LLM_NUM_HEADS = 6              # attention heads
LLM_NUM_LAYERS = 6             # transformer blocks
LLM_DROPOUT = 0.1              # dropout rate
LLM_LEARNING_RATE = 3e-4       # AdamW learning rate
LLM_BATCH_SIZE = 16            # training batch size
LLM_EPOCHS = 5                 # default training epochs
LLM_PARALLEL_WORKERS = 4       # 4-core parallelism for data loading & PDF extraction
