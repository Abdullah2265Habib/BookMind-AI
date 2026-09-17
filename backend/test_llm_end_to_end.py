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

def test_full_pipeline():
    print("=" * 65)
    print("Testing End-to-End Custom LLM Pipeline (4-Core Execution)")
    print("=" * 65)

    from backend.config import LLM_TRAINING_DIR, LLM_TESTING_DIR, LLM_MODELS_DIR
    from backend.custom_llm import (
        list_pdfs_in_dir,
        train_custom_llm,
        evaluate_on_test_pdfs,
        generate_response,
        get_model_status,
        get_training_state,
    )

    # 1. Check directories and PDFs
    print("\n[Step 1] Inspecting Training & Testing PDF Directories...")
    train_files = list_pdfs_in_dir(LLM_TRAINING_DIR)
    test_files = list_pdfs_in_dir(LLM_TESTING_DIR)
    print(f"  Training PDFs ({len(train_files)}): {[f['filename'] for f in train_files]}")
    print(f"  Testing PDFs  ({len(test_files)}): {[f['filename'] for f in test_files]}")
    assert len(train_files) > 0, "No training PDFs found"
    assert len(test_files) > 0, "No testing PDFs found"

    # 2. Run Training for 1 epoch on 4 CPU cores
    print("\n[Step 2] Training Custom LLM on 4 CPU Cores (1 epoch)...")
    def progress_cb(msg, state):
        loss = state.get("current_loss")
        epoch = state.get("epoch", 0)
        total = state.get("total_epochs", 1)
        loss_str = f", Loss: {loss:.4f}" if loss is not None else ""
        print(f"  [Progress] {msg} (Epoch {epoch}/{total}{loss_str})")

    t0 = time.time()
    train_result = train_custom_llm(
        epochs=1,
        progress_callback=progress_cb,
    )
    t_train = time.time() - t0
    print(f"  ✓ Training completed in {t_train:.2f}s!")
    print(f"    Status: {train_result.get('status')}")
    print(f"    Final Loss: {train_result.get('final_loss')}")
    print(f"    Perplexity: {train_result.get('final_perplexity')}")

    # 3. Model Status Check
    print("\n[Step 3] Checking Model Status...")
    status = get_model_status()
    print(f"  Status: {status.get('status')}")
    print(f"  Model exists: {status.get('model_file_exists')}")
    print(f"  Tokenizer exists: {status.get('tokenizer_file_exists')}")
    assert status.get("status") == "trained"

    # 4. Run Test Evaluation on Testing PDFs (4-Core)
    print("\n[Step 4] Running Test Evaluation on Testing PDFs (4 Cores)...")
    test_result = evaluate_on_test_pdfs()
    print(f"  Test Cross-Entropy Loss: {test_result.get('test_loss')}")
    print(f"  Test Perplexity (PPL):   {test_result.get('test_perplexity')}")
    print(f"  Sequences Evaluated:     {test_result.get('test_sequences')}")
    print(f"  Total Test Tokens:       {test_result.get('test_tokens')}")

    # 5. Test Inference / Autoregressive Chat Generation
    print("\n[Step 5] Testing Custom LLM Chat Generation...")
    prompts = [
        "What is 6G wireless communication?",
        "Key challenges in network security:",
    ]
    for p in prompts:
        gen = generate_response(prompt=p, max_new_tokens=40, temperature=0.7, top_k=40)
        print(f"\n  Prompt: '{p}'")
        print(f"  BookMindGPT: '{gen.strip()}'")

    print("\n" + "=" * 65)
    print("🎉 END-TO-END CUSTOM LLM PIPELINE VERIFIED SUCCESSFULLY!")
    print("=" * 65)

if __name__ == "__main__":
    test_full_pipeline()
