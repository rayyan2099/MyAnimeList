from pydantic import BaseModel
from typing import Optional, List


# Represents a single anime result returned by the API
class AnimeResult(BaseModel):
    anime_id: int
    title: str
    synopsis: str
    genres: str
    score: Optional[float] = None
    similarity: float


# Response for semantic search endpoint
class SearchResponse(BaseModel):
    query: str
    results: List[AnimeResult]


# Response for similar anime endpoint
class SimilarResponse(BaseModel):
    anime_id: int
    results: List[AnimeResult]