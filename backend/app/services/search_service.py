import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher
from app.core.config import settings


class SearchService:
    def __init__(self):
        # Load everything once when the app starts
        print("Loading FAISS index...")
        self.index = faiss.read_index(settings.FAISS_INDEX_PATH)

        print("Loading anime data...")
        self.df = pd.read_csv(settings.DATA_PATH)

        print("Loading embedding model...")
        self.model = SentenceTransformer(settings.MODEL_NAME)

        print("Loading embeddings...")
        self.embeddings = np.load(settings.EMBEDDINGS_PATH).astype("float32")

        print("All resources loaded successfully!")

    def semantic_search(self, query: str, top_k: int = settings.TOP_K):
        # Convert the search query text into a vector
        vec = self.model.encode([query])

        # Normalize so cosine similarity works correctly with FAISS
        vec = vec / np.linalg.norm(vec, axis=1, keepdims=True)
        vec = vec.astype("float32")

        # Search the FAISS index for the most similar vectors
        scores, indices = self.index.search(vec, top_k)

        # Build results list from the returned indices
        results = []
        for i, idx in enumerate(indices[0]):
            row = self.df.iloc[idx]
            results.append({
                "anime_id": int(row["anime_id"]),
                "title": row["title"],
                "synopsis": row["synopsis"],
                "genres": row["genres"],
                "score": row["score"] if not pd.isna(row["score"]) else None,
                "similarity": round(float(scores[0][i]), 3)
            })

        return results

    def similar_anime(self, anime_id: int, top_k: int = settings.TOP_K):
        # Find the anime in our dataframe by its MAL id
        matches = self.df[self.df["anime_id"] == anime_id]

        if matches.empty:
            return None

        query_idx = matches.index[0]
        query_title = self.df.iloc[query_idx]["title"]

        # Get the existing embedding for this anime (no re-encoding needed)
        query_vec = self.embeddings[query_idx:query_idx + 1].copy()
        faiss.normalize_L2(query_vec)

        # Search for more than top_k so we have room to filter out sequels/related
        scores, indices = self.index.search(query_vec, top_k + 20)

        results = []
        for i, idx in enumerate(indices[0]):
            # Skip the anime itself
            if idx == query_idx:
                continue

            row = self.df.iloc[idx]
            candidate_title = row["title"]

            # Filter out sequels and related entries e.g. "Death Note" and "Death Note: Rewrite"
            query_root = query_title.lower().split(":")[0].strip()
            candidate_root = candidate_title.lower().split(":")[0].strip()
            if query_root in candidate_title.lower() or candidate_root in query_title.lower():
                continue

            # Filter out titles that are too similar in name (ratio > 0.6 means very similar)
            ratio = SequenceMatcher(None, query_title.lower(), candidate_title.lower()).ratio()
            if ratio > 0.6:
                continue

            results.append({
                "anime_id": int(row["anime_id"]),
                "title": candidate_title,
                "synopsis": row["synopsis"],
                "genres": row["genres"],
                "score": row["score"] if not pd.isna(row["score"]) else None,
                "similarity": round(float(scores[0][i]), 3)
            })

            # Stop once we have enough results
            if len(results) == top_k:
                break

        return results


# Single instance that gets imported everywhere else
# This means the model and index are only loaded once
search_service = SearchService()