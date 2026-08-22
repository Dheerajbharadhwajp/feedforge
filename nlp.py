from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import threading
import warnings
from typing import Optional, List

warnings.filterwarnings("ignore", category=UserWarning)

analyzer = SentimentIntensityAnalyzer()
text_generator = None
emotion_classifier = None
generator_lock = threading.Lock()
model_lock = threading.Lock()

def analyze_sentiment(text: str) -> str:
    """Returns 'Positive', 'Negative', or 'Neutral' based on VADER compound score."""
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    if compound >= 0.05:
        return 'Positive'
    elif compound <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def _load_emotion_classifier():
    """Loads the distilroberta emotion model (called lazily in a background thread)."""
    global emotion_classifier
    from transformers import pipeline
    with model_lock:
        if emotion_classifier is None:
            print("Loading j-hartmann/emotion-english-distilroberta-base...")
            emotion_classifier = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=1,
                device=-1  # CPU
            )
            print("Emotion model loaded.")
    return emotion_classifier

def batch_analyze_emotions_transformer(texts: List[str], ratings: List[Optional[float]]) -> List[dict]:
    """
    Batch emotion analysis using distilroberta.
    Returns labels: anger, disgust, fear, joy, neutral, sadness, surprise.
    Applies rating-aware guardrail to prevent misclassifications.
    """
    clf = _load_emotion_classifier()
    try:
        truncated = [str(t)[:512] for t in texts]
        results = clf(truncated, batch_size=32)
        output = []
        for i, res in enumerate(results):
            top = res[0] if isinstance(res, list) else res
            label = top['label']
            score = float(top['score'])
            r = ratings[i] if ratings else None
            negative_labels = {'sadness', 'anger', 'fear', 'disgust'}
            if r is not None and r >= 4.0 and label in negative_labels:
                vader = analyze_sentiment(texts[i])
                if vader == 'Positive':
                    label, score = 'joy', 0.6
                elif vader == 'Neutral':
                    label, score = 'neutral', 0.6
            output.append({"label": label, "score": score})
        return output
    except Exception as e:
        print(f"Emotion batch failed: {e}")
        return [{"label": "neutral", "score": 0.0}] * len(texts)

def analyze_emotions(text: str, rating: Optional[float] = None) -> dict:
    """Single review emotion — uses VADER heuristic for instant response."""
    scores = analyzer.polarity_scores(str(text))
    compound = scores['compound']
    text_lower = text.lower()
    anger_words = ['horrible', 'disgusting', 'terrible', 'worst', 'awful', 'rude',
                   'unacceptable', 'outraged', 'furious', 'pathetic', 'ridiculous', 'never again']
    disgust_words = ['disgusting', 'gross', 'revolting', 'filthy', 'nasty', 'repulsive', 'vile']
    if any(w in text_lower for w in disgust_words) and compound < -0.2:
        return {"label": "disgust", "score": 0.7}
    if any(w in text_lower for w in anger_words) and compound < -0.2:
        return {"label": "anger", "score": 0.7}
    if compound >= 0.4:
        return {"label": "joy", "score": 0.8}
    elif compound >= 0.05:
        return {"label": "joy", "score": 0.5}
    elif compound <= -0.4:
        return {"label": "sadness", "score": 0.8}
    elif compound <= -0.05:
        return {"label": "sadness", "score": 0.5}
    else:
        return {"label": "neutral", "score": 0.6}

def get_autocomplete(text: str, max_new_tokens: int = 3) -> str:
    """Predicts next few words using distilgpt2."""
    global text_generator
    with generator_lock:
        if text_generator is None:
            from transformers import pipeline
            print("Loading distilgpt2 for auto-complete...")
            text_generator = pipeline('text-generation', model='distilgpt2')
            print("distilgpt2 loaded.")
    if not text.strip():
        return ""
    results = text_generator(
        text, max_new_tokens=max_new_tokens,
        num_return_sequences=1, do_sample=False, pad_token_id=50256
    )
    generated = results[0]['generated_text']
    if generated.startswith(text):
        return generated[len(text):].replace('\n', ' ')
    return ""
