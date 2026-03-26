from fastapi import APIRouter, HTTPException, Query
from app.services.search_service import search_service
from app.schemas import SearchResponse, SimilarResponse

# APIRouter is like a mini FastAPI app - we register it in main.py
router = APIRouter(
    prefix="/api/v1",
    tags=["search"]
)


@router.get("/search", response_model=SearchResponse)
def semantic_search(
    q: str = Query(..., min_length=1, description="Search query e.g. 'dark psychological thriller'"),
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    # Call the semantic search from our service
    results = search_service.semantic_search(query=q, top_k=top_k)

    return SearchResponse(query=q, results=results)


@router.get("/similar/{anime_id}", response_model=SimilarResponse)
def similar_anime(
    anime_id: int,
    top_k: int = Query(10, ge=1, le=50, description="Number of results to return")
):
    # Call the similarity search from our service
    results = search_service.similar_anime(anime_id=anime_id, top_k=top_k)

    # If anime_id doesn't exist in our data, return a 404
    if results is None:
        raise HTTPException(status_code=404, detail=f"Anime with id {anime_id} not found")

    return SimilarResponse(anime_id=anime_id, results=results)


@router.get("/health")
def health_check():
    # Simple endpoint to confirm the API is running
    return {"status": "ok"}