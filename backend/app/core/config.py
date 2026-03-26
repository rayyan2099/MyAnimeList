from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    FAISS_INDEX_PATH: str = str(BASE_DIR / "anime.index")
    EMBEDDINGS_PATH: str = str(BASE_DIR / "anime_embeddings.npy")
    DATA_PATH: str = str(BASE_DIR / "anime_clean.csv")
    MODEL_NAME: str = "all-MiniLM-L6-v2"
    TOP_K: int = 10 

    class Config:
        env_file = '.env'

settings = Settings()    
