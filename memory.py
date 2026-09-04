import pandas as pd
import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
import datetime
import re
import os
import time
import requests
from config import CSV_PATH, CHROMA_DB_PATH, COLLECTION_NAME
from nlp import analyze_sentiment, analyze_emotions

GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
GEMINI_EMBED_DIM = 768
GEMINI_EMBED_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/{GEMINI_EMBED_MODEL}:batchEmbedContents"
)
GEMINI_BATCH_LIMIT = 100  # max texts per batchEmbedContents call

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Calls Google's Gemini embeddings API instead of running a local model.
    Keeps the process's own memory footprint tiny, which matters on
    memory-constrained hosts (e.g. Render's 512Mi starter instance) where
    a local sentence-transformers/ONNX model easily exceeds the limit.
    """

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set; required for embeddings.")

    def __call__(self, input):
        texts = list(input)
        all_embeddings = []
        for i in range(0, len(texts), GEMINI_BATCH_LIMIT):
            chunk = texts[i:i + GEMINI_BATCH_LIMIT]
            all_embeddings.extend(self._embed_chunk(chunk))
        return all_embeddings

    def _embed_chunk(self, chunk, max_retries=5):
        body = {
            "requests": [
                {
                    "model": GEMINI_EMBED_MODEL,
                    "content": {"parts": [{"text": text[:2000]}]},
                    "outputDimensionality": GEMINI_EMBED_DIM,
                }
                for text in chunk
            ]
        }
        for attempt in range(max_retries):
            resp = requests.post(
                GEMINI_EMBED_URL,
                params={"key": self.api_key},
                json=body,
                timeout=30,
            )
            if resp.status_code == 429 and attempt < max_retries - 1:
                # Free tier is capped at 100 embed requests/minute; honor the
                # server's suggested retryDelay instead of a short fixed backoff.
                retry_delay = 60
                try:
                    for detail in resp.json().get("error", {}).get("details", []):
                        if "retryDelay" in detail:
                            retry_delay = int(float(detail["retryDelay"].rstrip("s"))) + 1
                except (ValueError, KeyError):
                    pass
                time.sleep(retry_delay)
                continue
            resp.raise_for_status()
            data = resp.json()
            return [item["values"] for item in data["embeddings"]]
        raise RuntimeError("Gemini embeddings API: exceeded retries")

    @staticmethod
    def name():
        return "gemini_text_embedding_004"

    def get_config(self):
        return {}

    @staticmethod
    def build_from_config(config):
        return GeminiEmbeddingFunction()

def parse_date_to_year_month(date_str: str):
    """Helper to extract year and month from diverse date formats."""
    if not date_str or pd.isna(date_str):
        return 2023, 1
        
    date_str = str(date_str).strip()
    try:
        # Match DD-MM-YYYY or DD/MM/YYYY
        match_num = re.search(r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', date_str)
        if match_num:
            day, month, year = int(match_num.group(1)), int(match_num.group(2)), int(match_num.group(3))
            # Basic swap protection if month > 12
            if month > 12 and day <= 12:
                month, day = day, month
            return year, month

        # Match "Month Day, Year" or "Month. Day, Year"
        match_text = re.search(r'([A-Za-z]+)\.?\s+\d+,\s+(\d{4})', date_str)
        if match_text:
            month_name = match_text.group(1)[:3]
            month = datetime.datetime.strptime(month_name, "%b").month
            year = int(match_text.group(2))
            return year, month

        # Match YYYY-MM-DD timestamp (e.g. 2025-11-27 08:15:26)
        match_iso = re.search(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', date_str)
        if match_iso:
            year, month = int(match_iso.group(1)), int(match_iso.group(2))
            return year, month
            
    except Exception:
        pass
    return 2023, 1

class MemoryBank:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.ef = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.ef,
            metadata={"description": "Agentic memory bank for customer reviews"}
        )

    def load_data(self, recreate=False, df=None):
        if recreate:
            try:
                self.client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=self.ef
            )
        
        # Check if already loaded
        if self.collection.count() > 0 and not recreate and df is None:
            return {"status": "success", "count": self.collection.count()}

        if df is None:
            if not os.path.exists(CSV_PATH):
                return {"status": "error", "message": "No data source available."}
            df = pd.read_csv(CSV_PATH)
        
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        emotion_counts = {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "neutral": 0}
        documents = []
        metadatas = []
        ids = []
        
        # Phase 1: Sequential Parsing (Fast)
        raw_entries = []
        for index, row in df.iterrows():
            review_text = row.get('Review', row.get('review', row.get('review_text', row.get('text', 'No Review Text'))))
            if pd.isna(review_text) or str(review_text).strip() in ["", "No Review Text"]:
                continue
            
            review_text = str(review_text)
            
            if 'Rating' in row:
                rating = float(row.get('Rating', 3.0))
            elif 'rating' in row:
                rating = float(row.get('rating', 3.0))
            else:
                rating = 3.0
                
            if 'Year' in row and 'Month' in row and not pd.isna(row['Year']) and not pd.isna(row['Month']):
                year, month = int(row['Year']), int(row['Month'])
            else:
                date_val = row.get('Date', row.get('Date_Cleaned', row.get('review_date', row.get('date', ''))))
                year, month = parse_date_to_year_month(date_val)
            
            location = str(row.get('Location', row.get('location', 'Unknown')))
            raw_entries.append({
                "text": review_text,
                "rating": rating,
                "year": year,
                "month": month,
                "location": location,
                "id": f"rev_{index}"
            })

        if not raw_entries:
            return {"status": "success", "count": 0}

        # Phase 2: High-Speed NLP Intelligence (VADER Sentiment + Heuristic Emotion)
        print(f"DEBUG: Processing {len(raw_entries)} reviews...")
        
        for entry in raw_entries:
            sentiment = analyze_sentiment(entry["text"])
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
            
            emotion_data = analyze_emotions(entry["text"], rating=entry["rating"])
            emotion = emotion_data["label"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
            metadata = {
                "rating": entry["rating"],
                "location": entry["location"],
                "year": entry["year"],
                "month": entry["month"],
                "review_length": len(entry["text"]),
                "sentiment": sentiment,
                "emotion": emotion,
                "review_id": entry["id"]
            }
            
            documents.append(entry["text"])
            metadatas.append(metadata)
            ids.append(entry["id"])
        
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size]
            )
            print(f"Inserted batch {i} to {min(i + batch_size, len(documents))}...")

        return {
            "status": "success",
            "vectors_embedded": len(documents),
            "sentiment_distribution": sentiment_counts,
            "emotion_distribution": emotion_counts
        }

    def get_emotion_trajectory(self):
        """Aggregates emotions over time (year-month) for longitudinal line charts."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", [])
        
        trajectory = {}
        for meta in metadatas:
            if 'year' not in meta or 'month' not in meta:
                continue
            key = f"{meta['year']}-{str(meta['month']).zfill(2)}"
            if key not in trajectory:
                trajectory[key] = {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "disgust": 0, "neutral": 0}
            
            emotion = meta.get("emotion", "neutral")
            if emotion in trajectory[key]:
                trajectory[key][emotion] += 1
            else:
                trajectory[key][emotion] = 1
            
        # Format for Recharts
        formatted = []
        for key in sorted(trajectory.keys()):
            entry = {"date": key}
            entry.update(trajectory[key])
            formatted.append(entry)
            
        return formatted


    def search_similar(self, query: str, n_results: int = 5, where: dict = None):
        """Returns similar reviews for the Synthesizer Agent."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where
        )
        return results

    def get_worst_reviews(self, limit=10):
        """Fetches low rating vectors for Owner Aggregate Insights."""
        if self.collection.count() == 0:
            return []
            
        results = self.collection.get(
            where={"rating": {"$lte": 2.0}},
            limit=limit,
            include=["documents"]
        )
        return results.get("documents", [])

memory_bank = MemoryBank()
