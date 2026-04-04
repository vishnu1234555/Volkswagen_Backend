import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .LLM import VolkswagenRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="VW RAG Production")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_pipeline: VolkswagenRAG | None = None


@app.on_event("startup")
def startup_event():
    global rag_pipeline
    try:
        rag_pipeline = VolkswagenRAG()
        logger.info("RAG Engine started with CORS enabled.")
    except Exception as e:
        logger.error("Startup failed: %s", e)
        rag_pipeline = None


class QueryRequest(BaseModel):
    question: str


@app.post("/retrieve")
def handle_query(request: QueryRequest):
    try:
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG engine is not ready (check configuration and logs).",
            )
        answer = rag_pipeline.generate_response(request.question)
        return {"status": "success", "question": request.question, "answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error: %s", e)
        raise HTTPException(status_code=500, detail=str(e)) from e
