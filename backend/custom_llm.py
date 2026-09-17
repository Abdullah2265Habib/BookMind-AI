"""
BookMind-AI: Custom LLM Built From Scratch
=============================================
A complete GPT-2 style language model implemented entirely from scratch
using only PyTorch primitives. No HuggingFace, no pretrained weights.

Architecture:
- Custom BPE Tokenizer (byte-pair encoding)
- Multi-Head Causal Self-Attention
- Stacked Transformer Decoder Blocks
- Learnable Token + Positional Embeddings
- Language Modeling Head

Training uses 4-core parallel execution for both PDF extraction
and DataLoader batch construction.
"""

import os
import re
import json
import math
import time
import pickle
import struct
import collections
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from backend.config import (
    LLM_TRAINING_DIR,
    LLM_TESTING_DIR,
    LLM_MODELS_DIR,
    LLM_VOCAB_SIZE,
    LLM_CONTEXT_LENGTH,
    LLM_EMBED_DIM,
    LLM_NUM_HEADS,
    LLM_NUM_LAYERS,
    LLM_DROPOUT,
    LLM_LEARNING_RATE,
    LLM_BATCH_SIZE,
    LLM_EPOCHS,
    LLM_PARALLEL_WORKERS,
)


# =====================================================================
#  PART 1: Custom BPE Tokenizer (Built From Scratch)
# =====================================================================

class BookMindTokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.
    Learns a subword vocabulary from raw text by iteratively merging
    the most frequent adjacent byte pairs.
    """

    def __init__(self, vocab_size: int = LLM_VOCAB_SIZE):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}
        self.inverse_vocab: Dict[bytes, int] = {}
        self._trained = False

        # Special tokens
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.unk_token_id = 2

    def _get_pair_counts(self, token_sequences: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Count frequency of all adjacent pairs in token sequences."""
        counts = collections.Counter()
        for seq in token_sequences:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def _merge_pair(self, token_sequences: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        """Replace all occurrences of pair with new_id in all sequences."""
        result = []
        for seq in token_sequences:
            new_seq = []
            i = 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(new_id)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            result.append(new_seq)
        return result

    def train(self, text: str, progress_callback: Optional[Callable] = None):
        """
        Train the BPE tokenizer on raw text.
        Starts with byte-level tokens (256) + 3 special tokens,
        then iteratively merges most frequent pairs.
        """
        # Initialize base vocabulary: 3 special tokens + 256 byte values
        num_special = 3
        self.vocab = {
            0: b"<PAD>",
            1: b"<EOS>",
            2: b"<UNK>",
        }
        for i in range(256):
            self.vocab[num_special + i] = bytes([i])

        # Convert text to byte-level token sequence
        text_bytes = text.encode("utf-8", errors="replace")
        token_sequences = [[num_special + b for b in text_bytes]]

        num_merges = self.vocab_size - num_special - 256
        next_id = num_special + 256

        for merge_step in range(num_merges):
            pair_counts = self._get_pair_counts(token_sequences)
            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < 2:
                break  # No pair appears enough to be worth merging

            # Perform merge
            self.merges[best_pair] = next_id
            # Build new vocab entry by concatenating the bytes of both parts
            new_bytes = self.vocab.get(best_pair[0], b"") + self.vocab.get(best_pair[1], b"")
            self.vocab[next_id] = new_bytes

            token_sequences = self._merge_pair(token_sequences, best_pair, next_id)
            next_id += 1

            if progress_callback and merge_step % 100 == 0:
                progress_callback(f"Tokenizer merge {merge_step}/{num_merges}", int(merge_step / num_merges * 100))

        # Build inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True

        if progress_callback:
            progress_callback("Tokenizer training complete", 100)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using learned BPE merges."""
        if not self._trained:
            raise RuntimeError("Tokenizer has not been trained yet.")

        num_special = 3
        tokens = [num_special + b for b in text.encode("utf-8", errors="replace")]

        # Apply merges in priority order (order they were learned)
        for pair, new_id in self.merges.items():
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    tokens = tokens[:i] + [new_id] + tokens[i + 2:]
                else:
                    i += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        byte_parts = []
        for tid in token_ids:
            if tid in (self.pad_token_id, self.eos_token_id):
                continue
            if tid in self.vocab:
                val = self.vocab[tid]
                if val in (b"<PAD>", b"<EOS>", b"<UNK>"):
                    continue
                byte_parts.append(val)
            else:
                byte_parts.append(b"?")
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def save(self, path: Path):
        """Save tokenizer state to disk."""
        state = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {str(k): v.hex() for k, v in self.vocab.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def load(self, path: Path):
        """Load tokenizer state from disk."""
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        self.vocab_size = state["vocab_size"]
        self.merges = {
            tuple(int(x) for x in k.split(",")): v
            for k, v in state["merges"].items()
        }
        self.vocab = {int(k): bytes.fromhex(v) for k, v in state["vocab"].items()}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._trained = True

    @property
    def is_trained(self) -> bool:
        return self._trained


# =====================================================================
#  PART 2: Transformer Model Architecture (Built From Scratch)
# =====================================================================

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal (Masked) Self-Attention built from scratch.
    Implements scaled dot-product attention with a causal mask to prevent
    attending to future tokens during autoregressive generation.
    """

    def __init__(self, embed_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # Combined QKV projection for efficiency
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # Registered as buffer so it moves to correct device and is not a parameter
        causal_mask = torch.tril(torch.ones(context_length, context_length))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch_size, seq_length, embed_dim

        # Compute Q, K, V in one shot
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # Each: (B, T, C)

        # Reshape for multi-head: (B, T, C) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        # Apply causal mask: set future positions to -inf
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)

        # Reshape back: (B, H, T, head_dim) -> (B, T, C)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation.
    FFN(x) = Linear(GELU(Linear(x)))
    Uses 4x expansion ratio as in the original Transformer / GPT-2.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2 = nn.Linear(4 * embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block:
    LayerNorm -> Causal Self-Attention -> Residual ->
    LayerNorm -> Feed-Forward Network -> Residual

    Uses Pre-Norm architecture (GPT-2 style) for training stability.
    """

    def __init__(self, embed_dim: int, num_heads: int, context_length: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, context_length, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm residual connection for attention
        x = x + self.attn(self.ln1(x))
        # Pre-norm residual connection for FFN
        x = x + self.ffn(self.ln2(x))
        return x


class BookMindGPT(nn.Module):
    """
    BookMindGPT: Complete GPT-2 style language model built from scratch.

    Architecture:
    - Token Embedding Layer (vocab_size -> embed_dim)
    - Positional Embedding Layer (context_length -> embed_dim)
    - N x TransformerBlock (causal self-attention + FFN)
    - Final LayerNorm
    - Language Modeling Head (embed_dim -> vocab_size)

    Parameters (~10M with default config):
    - 6 layers, 6 heads, 384 dim, 256 context
    """

    def __init__(
        self,
        vocab_size: int = LLM_VOCAB_SIZE,
        context_length: int = LLM_CONTEXT_LENGTH,
        embed_dim: int = LLM_EMBED_DIM,
        num_heads: int = LLM_NUM_HEADS,
        num_layers: int = LLM_NUM_LAYERS,
        dropout: float = LLM_DROPOUT,
    ):
        super().__init__()
        self.context_length = context_length
        self.embed_dim = embed_dim

        # Token and Positional Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks stack
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, context_length, dropout)
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.final_ln = nn.LayerNorm(embed_dim)

        # Language modeling head (projects back to vocabulary)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and LM head
        # This is a standard GPT-2 technique that improves performance
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Xavier/Glorot initialization for stable training."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            input_ids: (B, T) tensor of token IDs

        Returns:
            logits: (B, T, vocab_size) tensor of next-token predictions
        """
        B, T = input_ids.shape
        assert T <= self.context_length, f"Sequence length {T} exceeds context length {self.context_length}"

        # Create position indices
        positions = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)

        # Embed tokens and positions
        tok_emb = self.token_embedding(input_ids)      # (B, T, embed_dim)
        pos_emb = self.position_embedding(positions)     # (1, T, embed_dim)
        x = self.embed_dropout(tok_emb + pos_emb)       # (B, T, embed_dim)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.final_ln(x)

        # Project to vocabulary logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int = 1,
    ) -> torch.Tensor:
        """
        Autoregressive text generation with temperature scaling and top-k sampling.

        Args:
            input_ids: (1, T) seed token IDs
            max_new_tokens: maximum number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: number of top tokens to sample from
            eos_token_id: stop generation when this token is produced

        Returns:
            Generated token IDs tensor (1, T + generated)
        """
        self.eval()
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop to context length if needed
            context = generated[:, -self.context_length:]

            # Forward pass
            logits = self.forward(context)
            # Get logits for the last position only
            next_logits = logits[:, -1, :] / temperature  # (1, vocab_size)

            # Top-K filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                next_logits[next_logits < threshold] = float("-inf")

            # Sample from probability distribution
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            # Append generated token
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated
            if next_token.item() == eos_token_id:
                break

        return generated


# =====================================================================
#  PART 3: PDF Data Pipeline (4-Core Parallel Extraction)
# =====================================================================

def _extract_pdf_text_worker(pdf_path: str) -> str:
    """
    Worker function to extract all text from a single PDF.
    Runs inside ProcessPoolExecutor for 4-core parallelism.
    Uses PyMuPDF (fitz) for fast text extraction.
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        text_parts = []
        for page in doc:
            page_text = page.get_text("text") or ""
            cleaned = page_text.strip()
            if cleaned:
                text_parts.append(cleaned)
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        return f"[PDF Extraction Error: {str(e)}]"


def extract_all_pdfs_parallel(
    pdf_dir: Path,
    max_workers: int = LLM_PARALLEL_WORKERS,
    progress_callback: Optional[Callable] = None,
) -> str:
    """
    Extracts text from ALL PDF files in a directory using 4-core parallel execution.

    Args:
        pdf_dir: Directory containing PDF files
        max_workers: Number of parallel CPU cores (default: 4)
        progress_callback: Optional callback for progress updates

    Returns:
        Concatenated text from all PDFs
    """
    pdf_files = sorted([
        str(f) for f in pdf_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".pdf"
    ])

    if not pdf_files:
        return ""

    if progress_callback:
        progress_callback(f"Extracting text from {len(pdf_files)} PDFs using {max_workers} CPU cores...", 0)

    all_texts = []

    if len(pdf_files) == 1:
        # Single file: no need for multiprocessing overhead
        all_texts = [_extract_pdf_text_worker(pdf_files[0])]
    else:
        # 4-core parallel PDF extraction
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_extract_pdf_text_worker, pdf_files))
            all_texts = results

    if progress_callback:
        progress_callback(f"Extracted text from {len(pdf_files)} PDFs", 100)

    combined = "\n\n".join(t for t in all_texts if t and not t.startswith("[PDF"))
    return combined


class BookCorpusDataset(Dataset):
    """
    PyTorch Dataset for language modeling on tokenized book corpus.
    Creates sliding-window sequences of fixed context_length for causal LM training.
    """

    def __init__(self, token_ids: List[int], context_length: int = LLM_CONTEXT_LENGTH):
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        # Number of valid starting positions for sliding window
        # Each sample is (input[i:i+ctx], target[i+1:i+ctx+1])
        self.num_samples = max(0, len(token_ids) - context_length)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx
        end = start + self.context_length
        input_ids = self.token_ids[start:end]        # (context_length,)
        target_ids = self.token_ids[start + 1:end + 1]  # (context_length,) shifted by 1
        return input_ids, target_ids


# =====================================================================
#  PART 4: Training Engine (With Progress Callbacks)
# =====================================================================

# Global training state for progress polling
_TRAINING_STATE: Dict[str, Any] = {
    "status": "idle",  # idle, training, completed, failed
    "epoch": 0,
    "total_epochs": 0,
    "step": 0,
    "total_steps": 0,
    "train_loss": 0.0,
    "val_loss": 0.0,
    "val_perplexity": 0.0,
    "best_val_loss": float("inf"),
    "elapsed_seconds": 0,
    "message": "",
    "loss_history": [],  # List of {epoch, train_loss, val_loss, perplexity}
}


def get_training_state() -> Dict[str, Any]:
    """Returns a copy of the current training state for API polling."""
    return dict(_TRAINING_STATE)


def train_custom_llm(
    epochs: int = LLM_EPOCHS,
    progress_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Full training pipeline for the custom from-scratch LLM:

    1. Extract text from all training PDFs (4-core parallel)
    2. Train BPE tokenizer on the corpus
    3. Tokenize corpus and create DataLoader (4-core parallel workers)
    4. Train the BookMindGPT model with AdamW + cosine LR schedule
    5. Evaluate on test PDFs
    6. Save checkpoint

    Uses 4 CPU cores for parallelism throughout.
    """
    global _TRAINING_STATE

    _TRAINING_STATE["status"] = "training"
    _TRAINING_STATE["message"] = "Starting training pipeline..."
    _TRAINING_STATE["epoch"] = 0
    _TRAINING_STATE["total_epochs"] = epochs
    _TRAINING_STATE["loss_history"] = []
    start_time = time.time()

    def update_state(msg: str, **kwargs):
        _TRAINING_STATE["message"] = msg
        _TRAINING_STATE["elapsed_seconds"] = round(time.time() - start_time, 1)
        _TRAINING_STATE.update(kwargs)
        if progress_callback:
            progress_callback(msg, _TRAINING_STATE)

    try:
        # ---- Step 1: Extract training text (4-core parallel) ----
        update_state("Extracting text from training PDFs using 4 CPU cores...")
        train_text = extract_all_pdfs_parallel(
            LLM_TRAINING_DIR,
            max_workers=LLM_PARALLEL_WORKERS,
        )

        if not train_text or len(train_text) < 100:
            raise ValueError("Not enough training text. Please upload more PDFs to the training folder.")

        # ---- Step 2: Train BPE Tokenizer ----
        update_state(f"Training BPE tokenizer on {len(train_text):,} characters (vocab_size={LLM_VOCAB_SIZE})...")
        tokenizer = BookMindTokenizer(vocab_size=LLM_VOCAB_SIZE)
        tokenizer.train(train_text)

        # Save tokenizer
        tokenizer_path = LLM_MODELS_DIR / "tokenizer.json"
        tokenizer.save(tokenizer_path)
        update_state("BPE tokenizer trained and saved.")

        # ---- Step 3: Tokenize and Create Dataset ----
        update_state("Tokenizing training corpus...")
        all_token_ids = tokenizer.encode(train_text)
        update_state(f"Corpus tokenized: {len(all_token_ids):,} tokens")

        # Train/validation split (90/10)
        split_idx = int(len(all_token_ids) * 0.9)
        train_ids = all_token_ids[:split_idx]
        val_ids = all_token_ids[split_idx:]

        train_dataset = BookCorpusDataset(train_ids, context_length=LLM_CONTEXT_LENGTH)
        val_dataset = BookCorpusDataset(val_ids, context_length=LLM_CONTEXT_LENGTH)

        if len(train_dataset) < 1:
            raise ValueError("Training corpus too small. Need more text to create training sequences.")

        # DataLoader with 4-core parallel workers
        train_loader = DataLoader(
            train_dataset,
            batch_size=LLM_BATCH_SIZE,
            shuffle=True,
            num_workers=LLM_PARALLEL_WORKERS,
            pin_memory=False,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=LLM_BATCH_SIZE,
            shuffle=False,
            num_workers=LLM_PARALLEL_WORKERS,
            pin_memory=False,
            drop_last=False,
        ) if len(val_dataset) > 0 else None

        # ---- Step 4: Initialize Model ----
        update_state("Initializing BookMindGPT model from scratch...")
        actual_vocab = max(LLM_VOCAB_SIZE, max(all_token_ids) + 1)
        model = BookMindGPT(
            vocab_size=actual_vocab,
            context_length=LLM_CONTEXT_LENGTH,
            embed_dim=LLM_EMBED_DIM,
            num_heads=LLM_NUM_HEADS,
            num_layers=LLM_NUM_LAYERS,
            dropout=LLM_DROPOUT,
        )

        total_params = model.count_parameters()
        update_state(f"Model initialized: {total_params:,} parameters ({total_params/1e6:.1f}M)")

        # ---- Step 5: Training Loop ----
        optimizer = torch.optim.AdamW(model.parameters(), lr=LLM_LEARNING_RATE, weight_decay=0.01)

        # Cosine annealing learning rate schedule
        total_training_steps = len(train_loader) * epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_training_steps, eta_min=LLM_LEARNING_RATE * 0.1
        )

        _TRAINING_STATE["total_steps"] = total_training_steps
        best_val_loss = float("inf")

        for epoch in range(1, epochs + 1):
            _TRAINING_STATE["epoch"] = epoch
            model.train()
            epoch_loss = 0.0
            step_count = 0

            for batch_idx, (input_ids, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                # Forward pass
                logits = model(input_ids)  # (B, T, vocab_size)

                # Compute cross-entropy loss
                # Reshape: (B*T, vocab_size) vs (B*T,)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=0,  # ignore PAD token
                )

                # Backward pass
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                step_count += 1
                _TRAINING_STATE["step"] = (epoch - 1) * len(train_loader) + batch_idx + 1
                _TRAINING_STATE["train_loss"] = round(loss.item(), 4)

                if batch_idx % max(1, len(train_loader) // 10) == 0:
                    update_state(
                        f"Epoch {epoch}/{epochs} | Step {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                    )

            avg_train_loss = epoch_loss / max(step_count, 1)

            # ---- Validation ----
            val_loss = 0.0
            val_perplexity = 0.0
            if val_loader and len(val_dataset) > 0:
                model.eval()
                val_total_loss = 0.0
                val_steps = 0
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        val_logits = model(val_input)
                        v_loss = F.cross_entropy(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_target.view(-1),
                            ignore_index=0,
                        )
                        val_total_loss += v_loss.item()
                        val_steps += 1

                val_loss = val_total_loss / max(val_steps, 1)
                val_perplexity = math.exp(min(val_loss, 20))  # Cap to avoid overflow

            _TRAINING_STATE["val_loss"] = round(val_loss, 4)
            _TRAINING_STATE["val_perplexity"] = round(val_perplexity, 2)

            _TRAINING_STATE["loss_history"].append({
                "epoch": epoch,
                "train_loss": round(avg_train_loss, 4),
                "val_loss": round(val_loss, 4),
                "perplexity": round(val_perplexity, 2),
            })

            update_state(
                f"Epoch {epoch}/{epochs} complete | Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Perplexity: {val_perplexity:.1f}"
            )

            # Save best checkpoint
            if val_loss < best_val_loss or val_loader is None:
                best_val_loss = val_loss if val_loader else avg_train_loss
                _TRAINING_STATE["best_val_loss"] = round(best_val_loss, 4)

                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                    "vocab_size": actual_vocab,
                    "context_length": LLM_CONTEXT_LENGTH,
                    "embed_dim": LLM_EMBED_DIM,
                    "num_heads": LLM_NUM_HEADS,
                    "num_layers": LLM_NUM_LAYERS,
                    "total_params": total_params,
                }
                checkpoint_path = LLM_MODELS_DIR / "bookmind_gpt_best.pt"
                torch.save(checkpoint, str(checkpoint_path))

        # ---- Step 6: Test Evaluation (if test PDFs exist) ----
        test_result = None
        test_text = extract_all_pdfs_parallel(LLM_TESTING_DIR, max_workers=LLM_PARALLEL_WORKERS)
        if test_text and len(test_text) > 100:
            update_state("Evaluating model on test PDFs...")
            test_ids = tokenizer.encode(test_text)
            test_dataset = BookCorpusDataset(test_ids, context_length=LLM_CONTEXT_LENGTH)
            if len(test_dataset) > 0:
                test_loader = DataLoader(
                    test_dataset, batch_size=LLM_BATCH_SIZE, shuffle=False,
                    num_workers=LLM_PARALLEL_WORKERS, drop_last=False
                )
                model.eval()
                test_total_loss = 0.0
                test_steps = 0
                with torch.no_grad():
                    for t_input, t_target in test_loader:
                        t_logits = model(t_input)
                        t_loss = F.cross_entropy(
                            t_logits.view(-1, t_logits.size(-1)),
                            t_target.view(-1),
                            ignore_index=0,
                        )
                        test_total_loss += t_loss.item()
                        test_steps += 1

                test_loss = test_total_loss / max(test_steps, 1)
                test_perplexity = math.exp(min(test_loss, 20))
                test_result = {
                    "test_loss": round(test_loss, 4),
                    "test_perplexity": round(test_perplexity, 2),
                }
                update_state(f"Test evaluation: Loss={test_loss:.4f}, Perplexity={test_perplexity:.1f}")

        # ---- Save final metadata ----
        meta = {
            "status": "trained",
            "total_params": total_params,
            "vocab_size": actual_vocab,
            "context_length": LLM_CONTEXT_LENGTH,
            "embed_dim": LLM_EMBED_DIM,
            "num_heads": LLM_NUM_HEADS,
            "num_layers": LLM_NUM_LAYERS,
            "epochs_trained": epochs,
            "final_train_loss": round(avg_train_loss, 4),
            "final_val_loss": round(val_loss, 4),
            "final_val_perplexity": round(val_perplexity, 2),
            "best_val_loss": round(best_val_loss, 4),
            "training_time_seconds": round(time.time() - start_time, 1),
            "training_corpus_tokens": len(all_token_ids),
            "training_corpus_chars": len(train_text),
            "loss_history": _TRAINING_STATE["loss_history"],
            "test_result": test_result,
        }

        with open(LLM_MODELS_DIR / "model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        _TRAINING_STATE["status"] = "completed"
        update_state("Training complete! Model saved.", **meta)

        return meta

    except Exception as e:
        _TRAINING_STATE["status"] = "failed"
        _TRAINING_STATE["message"] = f"Training failed: {str(e)}"
        raise


def evaluate_on_test_pdfs() -> Dict[str, Any]:
    """
    Run evaluation on test PDFs using the trained model.
    Uses 4-core parallel PDF extraction and DataLoader.
    """
    # Load tokenizer
    tokenizer_path = LLM_MODELS_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found. Train the model first.")

    tokenizer = BookMindTokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint_path = LLM_MODELS_DIR / "bookmind_gpt_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Train the model first.")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    model = BookMindGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Extract test text
    test_text = extract_all_pdfs_parallel(LLM_TESTING_DIR, max_workers=LLM_PARALLEL_WORKERS)
    if not test_text or len(test_text) < 50:
        raise ValueError("Not enough test text. Upload PDFs to the testing folder.")

    test_ids = tokenizer.encode(test_text)
    test_dataset = BookCorpusDataset(test_ids, context_length=checkpoint["context_length"])

    if len(test_dataset) < 1:
        raise ValueError("Test corpus too small to create sequences.")

    test_loader = DataLoader(
        test_dataset, batch_size=LLM_BATCH_SIZE, shuffle=False,
        num_workers=LLM_PARALLEL_WORKERS, drop_last=False
    )

    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for input_ids, targets in test_loader:
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=0,
            )
            total_loss += loss.item()
            total_steps += 1

    avg_loss = total_loss / max(total_steps, 1)
    perplexity = math.exp(min(avg_loss, 20))

    return {
        "test_loss": round(avg_loss, 4),
        "test_perplexity": round(perplexity, 2),
        "test_sequences": len(test_dataset),
        "test_tokens": len(test_ids),
    }


# =====================================================================
#  PART 5: Inference / Chat Generation
# =====================================================================

def generate_response(
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """
    Generate text using the trained custom LLM.

    Args:
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-K sampling

    Returns:
        Generated text string
    """
    # Load tokenizer
    tokenizer_path = LLM_MODELS_DIR / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError("Tokenizer not found. Train the model first.")

    tokenizer = BookMindTokenizer()
    tokenizer.load(tokenizer_path)

    # Load model
    checkpoint_path = LLM_MODELS_DIR / "bookmind_gpt_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError("Model checkpoint not found. Train the model first.")

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    model = BookMindGPT(
        vocab_size=checkpoint["vocab_size"],
        context_length=checkpoint["context_length"],
        embed_dim=checkpoint["embed_dim"],
        num_heads=checkpoint["num_heads"],
        num_layers=checkpoint["num_layers"],
        dropout=0.0,  # No dropout during inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Encode prompt
    input_ids = tokenizer.encode(prompt)

    # Truncate to fit context window (leave room for generation)
    max_prompt_len = checkpoint["context_length"] - 10
    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    input_tensor = torch.tensor([input_ids], dtype=torch.long)

    # Generate
    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode generated tokens (exclude prompt)
    generated_ids = output_ids[0, len(input_ids):].tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


def get_model_status() -> Dict[str, Any]:
    """Returns the current status of the custom LLM (trained/untrained, metadata)."""
    meta_path = LLM_MODELS_DIR / "model_meta.json"
    checkpoint_path = LLM_MODELS_DIR / "bookmind_gpt_best.pt"
    tokenizer_path = LLM_MODELS_DIR / "tokenizer.json"

    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["model_file_exists"] = checkpoint_path.exists()
        meta["tokenizer_file_exists"] = tokenizer_path.exists()
        return meta

    return {
        "status": "untrained",
        "model_file_exists": checkpoint_path.exists(),
        "tokenizer_file_exists": tokenizer_path.exists(),
    }


def list_pdfs_in_dir(directory: Path) -> List[Dict[str, Any]]:
    """Lists all PDF files in a directory with their metadata."""
    pdfs = []
    if not directory.exists():
        return pdfs
    for f in sorted(directory.iterdir()):
        if f.is_file() and f.suffix.lower() == ".pdf":
            pdfs.append({
                "filename": f.name,
                "size_bytes": f.stat().st_size,
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })
    return pdfs
