from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tempfile, os
from services.ingestion_service import IngestionService
from services.langchain_service import LangChainService
from models.query_model import QueryRequest
from core.logger import logger
from core.errors import RetrievalError

app = FastAPI(title="Company RAG — Offline (Local Models)")

# Mount static directory for HTML and assets
app.mount("/static", StaticFiles(directory="static"), name="static")

ingestor = IngestionService()
lc_service = LangChainService()

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...), background_tasks: BackgroundTasks = None):
    """Supports multiple PDF or TXT files"""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tmp_paths = []
    try:
        # Save each uploaded file temporarily
        for file in files:
            suffix = os.path.splitext(file.filename)[-1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await file.read())
                tmp_paths.append((file.filename, tmp.name))

        def process_and_rebuild(pair):
            fname, path = pair
            try:
                logger.info(f"Background ingest: {fname} -> {path}")
                ingestor.ingest_file(path)
                try:
                    os.unlink(path)
                except Exception:
                    logger.warning(f"Failed to remove temp file: {path}")
                # Rebuild index after each file, or you can choose to rebuild once after all files
                lc_service.rebuild_index()
            except Exception as e:
                logger.error(f"Background ingestion error for {fname}: {e}")

        if background_tasks is not None:
            for pair in tmp_paths:
                background_tasks.add_task(process_and_rebuild, pair)
        else:
            for pair in tmp_paths:
                process_and_rebuild(pair)

        return {"status": "Ingestion started. Index will update shortly.", "files": [n for n, _ in tmp_paths]}

    except RetrievalError as e:
        logger.error(f"/ingest error: {e}")
        # cleanup temp files on error
        for _, p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"/ingest error: {e}")
        for _, p in tmp_paths:
            try: os.unlink(p)
            except Exception: pass
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(req: QueryRequest):
    try:
        res = lc_service.query(req.question, top_k=req.top_k)
        answer = res.get("result") or res.get("answer")
        sources = [getattr(d, "page_content", None) or str(d) for d in res.get("source_documents", [])]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"/query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild")
async def rebuild_index():
    try:
        lc_service.rebuild_index()
        return {"status": "reindexed"}
    except RetrievalError as e:
        logger.error(f"/rebuild error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"/rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def status():
    return {"status": "running", "docs_count": len(lc_service.index_manager.docs)}

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
