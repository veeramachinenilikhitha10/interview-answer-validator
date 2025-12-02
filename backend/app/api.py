from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from .rag.ingest import build_index
from .rag.retrieval import retrieve

router = APIRouter()

class QueryBody(BaseModel):
    query: str
    top_k: int = 4

@router.get('/health')
async def health():
    return {'status': 'ok'}

@router.post('/ingest')
async def ingest_endpoint():
    try:
        res = build_index()
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/query')
async def query_endpoint(body: QueryBody = Body(...)):
    try:
        results = await retrieve(body.query, top_k=body.top_k)
        return {'results': results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ---- Evaluation Endpoint (appended) ----
from pydantic import BaseModel
from .rag.retrieval import retrieve
from .rag.llm_wrapper import call_llm_for_evaluation

class EvalReq(BaseModel):
    candidate_answer: str
    question: str = "Why do you want this job?"
    top_k: int = 4

@router.post("/evaluate")
async def evaluate(req: EvalReq):
    try:
        passages = await retrieve(req.question, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval error: {e}")

    res = await call_llm_for_evaluation(
        req.candidate_answer,
        req.question,
        passages
    )
    return res

