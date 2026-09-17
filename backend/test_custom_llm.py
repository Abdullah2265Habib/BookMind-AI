import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def test_scratch_llm():
    print("=" * 60)
    print("Testing BookMindGPT Custom LLM (Built From Scratch)")
    print("=" * 60)

    import torch
    from backend.custom_llm import (
        BookMindTokenizer,
        BookMindGPT,
        CausalSelfAttention,
        TransformerBlock,
        generate_response,
        get_model_status,
    )

    print(f"PyTorch Version: {torch.__version__}")
    print(f"PyTorch CPU Threads: {torch.get_num_threads()}")

    # 1. Test Tokenizer
    print("\n[1/4] Testing Custom BPE Tokenizer...")
    sample_corpus = (
        "BookMind-AI is a high-performance generic book-to-AI pipeline. "
        "It features hierarchical semantic chunking, FastEmbed ONNX embeddings, "
        "BM25 lexical search with Reciprocal Rank Fusion, and custom LLM training from scratch. "
        "Next generation 6G wireless communication networks rely on terahertz frequencies, "
        "reconfigurable intelligent surfaces, ultra-massive MIMO, and artificial intelligence."
    )
    tokenizer = BookMindTokenizer(vocab_size=350)
    t0 = time.time()
    tokenizer.train(sample_corpus)
    t_train = time.time() - t0
    print(f"  BPE trained in {t_train:.3f}s. Vocab size: {len(tokenizer.vocab)}")

    test_sentence = "BookMind-AI 6G wireless networks."
    encoded = tokenizer.encode(test_sentence)
    decoded = tokenizer.decode(encoded)
    print(f"  Original: '{test_sentence}'")
    print(f"  Encoded token IDs ({len(encoded)}): {encoded}")
    print(f"  Decoded:  '{decoded}'")
    assert test_sentence == decoded, f"Mismatch: '{test_sentence}' != '{decoded}'"
    print("  ✓ Tokenizer round-trip test passed!")

    # 2. Test Architecture & Forward Pass
    print("\n[2/4] Testing Transformer Architecture & Forward Pass...")
    vocab_sz = len(tokenizer.vocab)
    ctx_len = 64
    embed_dim = 96
    n_heads = 4
    n_layers = 3

    model = BookMindGPT(
        vocab_size=vocab_sz,
        context_length=ctx_len,
        embed_dim=embed_dim,
        num_heads=n_heads,
        num_layers=n_layers,
        dropout=0.0,
    )
    total_params = model.count_parameters()
    print(f"  Mini Model Parameters: {total_params:,}")

    # Random batch of shape (batch_size=2, seq_len=16)
    x = torch.randint(0, vocab_sz, (2, 16))
    targets = torch.randint(0, vocab_sz, (2, 16))

    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, vocab_sz), targets.view(-1))
    print(f"  Input shape:   {x.shape}")
    print(f"  Logits shape:  {logits.shape}")
    print(f"  Initial Loss:  {loss.item():.4f}")
    assert logits.shape == (2, 16, vocab_sz), f"Unexpected shape: {logits.shape}"
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ Forward pass & loss computation passed!")

    # 3. Test Mini Overfitting Step (Loss must decrease)
    print("\n[3/4] Testing 5 Optimization Steps (verifying gradient flow)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    initial_loss = loss.item()
    for step in range(5):
        optimizer.zero_grad()
        step_logits = model(x)
        step_loss = torch.nn.functional.cross_entropy(step_logits.view(-1, vocab_sz), targets.view(-1))
        step_loss.backward()
        optimizer.step()

    final_loss = step_loss.item()
    print(f"  Loss before: {initial_loss:.4f} -> after 5 steps: {final_loss:.4f}")
    assert final_loss < initial_loss, "Loss did not decrease during optimization"
    print("  ✓ Backpropagation & gradient flow verified!")

    # 4. Test Autoregressive Generation
    print("\n[4/4] Testing Autoregressive Generation...")
    start_tokens = torch.tensor([[encoded[0]]], dtype=torch.long)
    generated = model.generate(start_tokens, max_new_tokens=15, temperature=0.8, top_k=20)
    print(f"  Generated sequence length: {generated.shape[1]}")
    gen_text = tokenizer.decode(generated[0].tolist())
    print(f"  Generated text: '{gen_text}'")
    print("  ✓ Autoregressive generation passed!")

    print("\n" + "=" * 60)
    print("🎉 ALL CUSTOM LLM FROM SCRATCH UNIT TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    test_scratch_llm()
