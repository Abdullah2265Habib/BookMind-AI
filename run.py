import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting BookMind-AI: Generic Book-to-AI Pipeline & RAG System")
    print("Web Interface & RAG Server: http://127.0.0.1:8000")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
