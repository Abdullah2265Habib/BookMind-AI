import os
import sys
import uuid
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Workspace setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from backend.config import (
    UPLOADS_DIR,
    BOOKS_DIR,
    FRONTEND_DIR,
    NUM_WORKERS,
    EMBEDDING_MODEL_NAME,
    DEFAULT_LLM_PROVIDER,
    GEMINI_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    LLM_TRAINING_DIR,
    LLM_TESTING_DIR,
)
from backend.pipeline import BookPipeline
from backend.vector_store import BookVectorStore
from backend.rag_engine import RAGEngine

app = FastAPI(
    title="BookMind-AI: Generic Book-to-AI Pipeline & RAG System",
    description="Automated book PDF ingestion, 4-core CPU parallel parsing, vector knowledge base, and citation-grounded RAG chatbot",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job registry for tracking upload/processing progress
JOBS: Dict[str, Dict[str, Any]] = {}
rag_engine = RAGEngine()

class ChatRequest(BaseModel):
    book_id: str
    query: str
    provider: Optional[str] = "gemini"
    api_key: Optional[str] = None
    model: Optional[str] = None

class SettingsUpdate(BaseModel):
    provider: str
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

class CustomLLMChatRequest(BaseModel):
    query: str
    book_id: Optional[str] = None

# Global runtime settings
RUNTIME_SETTINGS = {
    "provider": DEFAULT_LLM_PROVIDER,
    "gemini_api_key": GEMINI_API_KEY,
    "openai_api_key": OPENAI_API_KEY,
    "groq_api_key": GROQ_API_KEY
}

@app.get("/api/status")
def get_system_status():
    """System hardware and engine telemetry."""
    import psutil
    return {
        "status": "online",
        "cpu_count_total": os.cpu_count(),
        "parallel_worker_cores": NUM_WORKERS,
        "cpu_usage_percent": psutil.cpu_percent(interval=None),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "active_provider": RUNTIME_SETTINGS["provider"],
        "has_gemini_key": bool(RUNTIME_SETTINGS["gemini_api_key"]),
        "has_openai_key": bool(RUNTIME_SETTINGS["openai_api_key"]),
        "has_groq_key": bool(RUNTIME_SETTINGS["groq_api_key"])
    }

@app.get("/api/books")
def list_books():
    """Returns all processed books and their knowledge bases."""
    books = []
    if not BOOKS_DIR.exists():
        return books

    for item in BOOKS_DIR.iterdir():
        if item.is_dir():
            meta_path = item / "metadata.json"
            if meta_path.exists():
                try:
                    import json
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        books.append(meta)
                except Exception:
                    pass
    return sorted(books, key=lambda b: b.get("title", ""))

@app.get("/api/books/{book_id}")
def get_book_details(book_id: str):
    """Returns complete metadata and chunk outline for a book."""
    store = BookVectorStore(book_id)
    if not store.is_ready():
        raise HTTPException(status_code=404, detail="Book knowledge base not found")
    
    # Return metadata and lightweight sample of chunks
    return {
        "metadata": store.metadata,
        "chunks_sample": store.chunks[:20],
        "total_chunks": len(store.chunks)
    }

@app.get("/api/books/{book_id}/pdf")
def get_book_pdf(book_id: str):
    """Serves the original PDF file for embedded viewer."""
    store = BookVectorStore(book_id)
    pdf_path = store.book_dir / "document.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Original PDF file not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")

def _run_ingestion_job(job_id: str, file_path: str, book_id: str):
    """Background worker for PDF-to-AI processing."""
    try:
        def update_progress(stage: str, percent: int, data: Dict[str, Any]):
            JOBS[job_id]["stage"] = stage
            JOBS[job_id]["progress"] = percent
            if data:
                JOBS[job_id]["data"].update(data)

        pipeline = BookPipeline(num_workers=NUM_WORKERS)
        result = pipeline.process_book(
            pdf_path=file_path,
            book_id=book_id,
            progress_callback=update_progress
        )
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["result"] = result
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

@app.post("/api/upload")
async def upload_book(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    custom_title: Optional[str] = Form(None)
):
    """Upload any PDF book and launch 4-core parallel processing."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    safe_name = "".join(c if c.isalnum() else "_" for c in Path(file.filename).stem).lower()
    book_id = f"book_{safe_name}_{uuid.uuid4().hex[:4]}"

    dest_file = UPLOADS_DIR / f"{book_id}_{file.filename}"
    with open(dest_file, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOBS[job_id] = {
        "job_id": job_id,
        "book_id": book_id,
        "filename": file.filename,
        "status": "processing",
        "progress": 5,
        "stage": "Uploaded PDF. Initializing 4-core pipeline...",
        "data": {}
    }

    background_tasks.add_task(_run_ingestion_job, job_id, str(dest_file), book_id)

    return {
        "job_id": job_id,
        "book_id": book_id,
        "message": "Upload accepted. Ingestion started in background."
    }

@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    """Check status of ingestion job."""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]

@app.post("/api/process-preset")
def process_preset(background_tasks: BackgroundTasks):
    """Processes the test PDF located in PDF/ folder."""
    preset_pdf = BASE_DIR / "PDF" / "next generation computer networks.pdf"
    if not preset_pdf.exists():
        raise HTTPException(status_code=404, detail="Preset PDF not found")

    job_id = f"job_preset_{uuid.uuid4().hex[:6]}"
    book_id = "book_6g_networks"

    JOBS[job_id] = {
        "job_id": job_id,
        "book_id": book_id,
        "filename": "next generation computer networks.pdf",
        "status": "processing",
        "progress": 5,
        "stage": "Initializing 4-core pipeline for sample paper...",
        "data": {}
    }

    background_tasks.add_task(_run_ingestion_job, job_id, str(preset_pdf), book_id)

    return {"job_id": job_id, "book_id": book_id}

@app.post("/api/chat")
async def chat_query(req: ChatRequest):
    """
    RAG Chat query endpoint with strict citation grounding.
    """
    store = BookVectorStore(req.book_id)
    if not store.is_ready():
        raise HTTPException(status_code=404, detail="Selected book knowledge base is not ready")

    # Determine provider credentials
    provider = req.provider or RUNTIME_SETTINGS["provider"]
    api_key = req.api_key
    if not api_key:
        if provider == "gemini":
            api_key = RUNTIME_SETTINGS["gemini_api_key"]
        elif provider == "openai":
            api_key = RUNTIME_SETTINGS["openai_api_key"]
        elif provider == "groq":
            api_key = RUNTIME_SETTINGS["groq_api_key"]

    result = await rag_engine.answer(
        book_store=store,
        query=req.query,
        provider=provider,
        api_key=api_key,
        model=req.model
    )

    return result

@app.post("/api/settings")
def update_settings(settings: SettingsUpdate):
    """Update runtime LLM settings."""
    RUNTIME_SETTINGS["provider"] = settings.provider
    if settings.gemini_api_key is not None:
        RUNTIME_SETTINGS["gemini_api_key"] = settings.gemini_api_key
    if settings.openai_api_key is not None:
        RUNTIME_SETTINGS["openai_api_key"] = settings.openai_api_key
    if settings.groq_api_key is not None:
        RUNTIME_SETTINGS["groq_api_key"] = settings.groq_api_key
    return {"status": "success", "settings": {
        "provider": RUNTIME_SETTINGS["provider"],
        "has_gemini_key": bool(RUNTIME_SETTINGS["gemini_api_key"]),
        "has_openai_key": bool(RUNTIME_SETTINGS["openai_api_key"]),
        "has_groq_key": bool(RUNTIME_SETTINGS["groq_api_key"])
    }}


# =====================================================================
#  Custom From-Scratch LLM API Endpoints
# =====================================================================

@app.get("/api/llm/status")
def get_llm_status():
    """Returns the current status of the custom from-scratch LLM."""
    from backend.custom_llm import get_model_status, get_training_state
    status = get_model_status()
    training = get_training_state()
    status["training_state"] = training
    return status

@app.post("/api/llm/upload-training")
async def upload_training_pdf(file: UploadFile = File(...)):
    """Upload a PDF to the training folder for custom LLM training."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = LLM_TRAINING_DIR / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": file.filename,
        "folder": "training",
        "size_bytes": dest.stat().st_size,
    }

@app.post("/api/llm/upload-testing")
async def upload_testing_pdf(file: UploadFile = File(...)):
    """Upload a PDF to the testing folder for custom LLM evaluation."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest = LLM_TESTING_DIR / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "status": "success",
        "filename": file.filename,
        "folder": "testing",
        "size_bytes": dest.stat().st_size,
    }

@app.get("/api/llm/training-files")
def list_training_files():
    """Lists all PDF files in the training folder."""
    from backend.custom_llm import list_pdfs_in_dir
    return list_pdfs_in_dir(LLM_TRAINING_DIR)

@app.get("/api/llm/testing-files")
def list_testing_files():
    """Lists all PDF files in the testing folder."""
    from backend.custom_llm import list_pdfs_in_dir
    return list_pdfs_in_dir(LLM_TESTING_DIR)

@app.delete("/api/llm/training-files/{filename}")
def delete_training_file(filename: str):
    """Delete a PDF from the training folder."""
    file_path = LLM_TRAINING_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"status": "deleted", "filename": filename}

@app.delete("/api/llm/testing-files/{filename}")
def delete_testing_file(filename: str):
    """Delete a PDF from the testing folder."""
    file_path = LLM_TESTING_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    file_path.unlink()
    return {"status": "deleted", "filename": filename}

@app.post("/api/llm/train")
def start_llm_training(background_tasks: BackgroundTasks):
    """Launch custom LLM training in background (4-core parallel)."""
    from backend.custom_llm import get_training_state, train_custom_llm

    current_state = get_training_state()
    if current_state["status"] == "training":
        raise HTTPException(status_code=409, detail="Training is already in progress.")

    # Check training PDFs exist
    training_pdfs = list((LLM_TRAINING_DIR).iterdir())
    training_pdfs = [f for f in training_pdfs if f.suffix.lower() == ".pdf"]
    if not training_pdfs:
        raise HTTPException(status_code=400, detail="No training PDFs found. Upload PDFs first.")

    def run_training():
        try:
            train_custom_llm()
        except Exception as e:
            print(f"[CustomLLM] Training error: {e}")

    background_tasks.add_task(run_training)

    return {
        "status": "started",
        "message": "Training started in background using 4 CPU cores.",
        "training_pdfs": len(training_pdfs),
    }

@app.get("/api/llm/train-status")
def get_llm_train_status():
    """Returns real-time training progress (epoch, loss, perplexity)."""
    from backend.custom_llm import get_training_state
    return get_training_state()

@app.post("/api/llm/test")
def run_llm_test():
    """Run evaluation on testing PDFs using the trained model."""
    from backend.custom_llm import evaluate_on_test_pdfs
    try:
        result = evaluate_on_test_pdfs()
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/llm/chat")
async def chat_with_custom_llm(req: CustomLLMChatRequest):
    """Chat directly with the custom from-scratch LLM."""
    from backend.custom_llm import generate_response, get_model_status

    status = get_model_status()
    if status.get("status") != "trained":
        raise HTTPException(
            status_code=400,
            detail="Custom LLM is not trained yet. Train the model first."
        )

    # If book_id provided, use RAG-enhanced generation
    if req.book_id:
        store = BookVectorStore(req.book_id)
        if store.is_ready():
            result = await rag_engine.answer(
                book_store=store,
                query=req.query,
                provider="custom_llm",
            )
            return result

    # Direct generation without RAG
    try:
        generated = generate_response(
            prompt=req.query,
            max_new_tokens=200,
            temperature=0.7,
            top_k=40,
        )
        return {
            "answer": generated,
            "provider": "custom_llm",
            "citations": [],
            "passages_count": 0,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


# Mount frontend directory for static UI serving
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "BookMind-AI backend is running. Frontend index.html not found."})

