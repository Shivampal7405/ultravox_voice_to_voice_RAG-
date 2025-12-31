"""
RAG Tool Server
FastAPI endpoint for Ultravox RAG tool integration
"""
import asyncio
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from loguru import logger
import uvicorn

from rag.rag_service import RAGService
from core.config import get_rag_settings


# FastAPI app
app = FastAPI(
    title="Voice RAG Tool Server",
    description="RAG search endpoint for Ultravox voice integration",
    version="1.0.0"
)

# CORS for Ultravox
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG service (lazy init)
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance"""
    global _rag_service
    if _rag_service is None:
        settings = get_rag_settings()
        _rag_service = RAGService(
            groq_api_key=settings.groq_api_key,
            milvus_uri=settings.milvus_uri,
            embedding_model=settings.embedding_model,
            groq_model_name=settings.groq_model
        )
    return _rag_service


class SearchRequest(BaseModel):
    """RAG search request from Ultravox"""
    query: str


class SearchResult(BaseModel):
    """RAG search result for Ultravox"""
    result: str
    sources: int = 0


@app.post("/searchKnowledge")
async def search_knowledge(request: SearchRequest) -> SearchResult:
    """
    Search the knowledge base
    
    This endpoint is called by Ultravox when the agent uses the searchKnowledge tool.
    """
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"[RAG TOOL] Query: {query}")
    
    try:
        rag = get_rag_service()
        settings = get_rag_settings()
        
        # Retrieve context (fast, no LLM generation)
        context = rag.retrieve_context(
            query_text=query,
            top_k=settings.top_k,
            collection_name=settings.collection
        )
        
        if not context or context.startswith("Error"):
            logger.warning(f"[RAG TOOL] No results for: {query}")
            return SearchResult(
                result="No relevant information found in the knowledge base.",
                sources=0
            )
        
        # Count sources
        source_count = context.count("[Source")
        
        logger.info(f"[RAG TOOL] Found {source_count} sources")
        
        return SearchResult(
            result=context,
            sources=source_count
        )
        
    except Exception as e:
        logger.error(f"[RAG TOOL] Error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG search failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "rag-tool-server"}


@app.on_event("startup")
async def startup():
    """Initialize on startup"""
    logger.info("[RAG TOOL] Server starting...")
    try:
        # Pre-initialize RAG service
        get_rag_service()
        logger.info("[RAG TOOL] RAG service initialized")
    except Exception as e:
        logger.error(f"[RAG TOOL] Failed to initialize RAG: {e}")
        # Don't fail startup, will retry on first request


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    global _rag_service
    if _rag_service:
        logger.info("[RAG TOOL] Disconnecting from Milvus...")
        _rag_service.vector_store.disconnect()
        _rag_service = None


def run_server(host: str = "0.0.0.0", port: int = 8001):
    """Run the RAG tool server"""
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    settings = get_rag_settings()
    run_server(port=settings.tool_port)
