from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.search import router

# Create the FastAPI app
app = FastAPI(
    title="Anime Recommendation API",
    description="Semantic search and anime similarity powered by FAISS and Sentence Transformers",
    version="1.0.0"
)

# CORS middleware allows your frontend (HTML/JS) to call this API
# Without this, browsers block requests from a different origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register the search router - all routes defined in search.py are now active
app.include_router(router)


@app.get("/")
def root():
    return {"message": "Anime Recommendation API is running"}
