from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import asyncio
import uuid
import pandas as pd
import io
import json

import config
from memory import memory_bank
from graph import feedback_pipeline
from nlp import get_autocomplete, analyze_sentiment, analyze_emotions

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="FeedbackForge API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from graph_memory import extract_graph_from_df, get_graph_data

@app.on_event("startup")
async def startup_event():
    # Runs in a background thread so the server starts accepting requests
    # immediately instead of blocking on the (rate-limited) embedding pass.
    asyncio.create_task(asyncio.to_thread(memory_bank.load_data))
    # Graph is built on-demand during CSV upload, not on startup

class ReviewRequest(BaseModel):
    review_text: str
    rating: float = 3.0

class ReviewResponse(BaseModel):
    review_id: str
    category: Optional[str]
    urgency_score: Optional[float]
    root_cause: Optional[str]
    final_output: Optional[str]
    sentiment: Optional[str]
    similar_reviews: Optional[List[str]]

class AutocompleteRequest(BaseModel):
    text: str

class ChatMessage(BaseModel):
    role: str
    content: str

class OwnerChatRequest(BaseModel):
    query: str
    topic_context: dict
    messages: List[ChatMessage]

@app.post("/api/analyze-review", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    review_id = f"mock_{uuid.uuid4().hex[:8]}"
    initial_state = {
        "review_id": review_id,
        "review_text": request.review_text,
        "metadata": {
            "rating": request.rating,
            "review_length": len(request.review_text)
        }
    }
    
    try:
        result_state = feedback_pipeline.invoke(initial_state)
        # Real-time NLP Enrichment (Sentiment Only)
        sentiment = analyze_sentiment(request.review_text)
        
        return ReviewResponse(
            review_id=review_id,
            category=result_state.get("category"),
            urgency_score=result_state.get("urgency_score"),
            root_cause=result_state.get("root_cause"),
            final_output=result_state.get("final_output"),
            sentiment=sentiment,
            similar_reviews=result_state.get("similar_reviews", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/memory/similar")
async def get_similar_reviews(query: str, limit: int = 5):
    try:
        results = memory_bank.search_similar(query=query, n_results=limit)
        similar_texts = []
        if results and "documents" in results and len(results["documents"]) > 0:
            similar_texts = results["documents"][0]
        return {"query": query, "similar_reviews": similar_texts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        result = memory_bank.load_data(recreate=True, df=df)
        
        # Concurrently build the Node-Edge Knowledge Graph
        extract_graph_from_df(df)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph-data")
async def fetch_graph_data():
    try:
        return get_graph_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/emotion-analytics")
async def fetch_emotion_analytics():
    try:
        trajectory = memory_bank.get_emotion_trajectory()
        return {"trajectory": trajectory}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/autocomplete")
async def autocomplete(request: AutocompleteRequest):
    try:
        suggestion = get_autocomplete(request.text)
        return {"suggestion": suggestion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/owner-insights")
async def get_owner_insights():
    try:
        worst_reviews = memory_bank.get_worst_reviews(limit=15)
        print(f"DEBUG: Found {len(worst_reviews)} worst reviews for analysis.")
        
        if not worst_reviews:
            return {"insights": []}

        # ✅ Proper newline (not escaped)
        context_str = "\n".join([f"- {r}" for r in worst_reviews])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an Executive Business Strategist analyzing negative customer feedback.\n"
             "Analyze the provided reviews and identify the top 3 systemic issues.\n\n"
             "Rules:\n"
             "- Focus ONLY on negative sentiment\n"
             "- Base insights strictly on the provided reviews\n"
             "- Identify recurring patterns\n"
             "- Do NOT use external knowledge\n\n"
             "Return a JSON array with EXACTLY 3 objects.\n"
             "Each object MUST contain:\n"
             "1) \"question\"\n"
             "2) \"explanation\"\n"
             "3) \"solution\" (This MUST be a highly detailed, step-by-step remediation plan with at least 3 concrete steps)\n"
             "4) \"urgency_score\" (A float from 1.0 to 10.0 indicating the severity of the issue based on the reviews)\n\n"
             "Output MUST be valid JSON. No extra text."),
            
            ("user", "Reviews:\n{context}")
        ])

        llm = ChatGroq(
            model=config.GROQ_STRATEGIST_MODEL,
            temperature=0.1
        )

        try:
            chain = prompt | llm | StrOutputParser()
            result_str = chain.invoke({"context": context_str})

            # ✅ SAFE JSON PARSING
            try:
                # First try direct parse
                insights = json.loads(result_str)
            except json.JSONDecodeError:
                # Fallback to regex extraction
                import re
                json_match = re.search(r'\[\s*\{.*\}\s*\]', result_str, re.DOTALL)
                if json_match:
                    try:
                        insights = json.loads(json_match.group(0))
                    except:
                        # Last resort: try just finding any [ ... ] block
                        json_match = re.search(r'\[.*\]', result_str, re.DOTALL)
                        if json_match:
                            insights = json.loads(json_match.group(0))
                        else:
                            raise ValueError("LLM did not return valid JSON array")
                else:
                    raise ValueError("LLM did not return valid JSON structure")

            return {"insights": insights}
        except Exception as inner_e:
            print(f"ERROR in Owner Insights Groq call: {str(inner_e)}")
            return {"insights": [], "error": str(inner_e)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/owner-chat")
async def owner_chat(request: OwnerChatRequest):
    try:
        # Dynamic RAG Query with Emotion Intelligence
        similar_results = memory_bank.search_similar(query=request.query, n_results=15)
        historical_context = []
        if similar_results and "documents" in similar_results and len(similar_results["documents"]) > 0:
            docs = similar_results["documents"][0]
            metas = similar_results["metadatas"][0]
            for d, m in zip(docs, metas):
                historical_context.append(f"[SENTIMENT: {m.get('sentiment')}] {d}")
        
        # Format Memory
        history_str = "\n".join([f"{m.role.upper()}: {m.content}" for m in request.messages[-5:]]) # Last 5 turns
        docs_str = "\n".join([f"- {d}" for d in historical_context])
        
        # Escape any braces in the dynamic strings to prevent LangChain from treating them as variables
        docs_str_escaped = docs_str.replace("{", "{{").replace("}", "}}")
        history_str_escaped = history_str.replace("{", "{{").replace("}", "}}")
        solution_escaped = json.dumps(request.topic_context.get('solution', {})).replace("{", "{{").replace("}", "}}")
        focus_q = request.topic_context.get('question', '').replace("{", "{{").replace("}", "}}")

        system_prompt = f"""You are an aggressive, visionary Executive SLM powered by the cutting-edge Llama 3 architecture.
You advise the company owner. You possess COMPLETE dataset awareness of historic reviews.

YOUR PRIME DIRECTIVES:
1. Provide crazy, highly-specialized, disruptive ideas for the company's future. 
2. Absolutely DO NOT give safe, generic, or broad corporate advice.
3. Leverage the exact specific mechanical details and emotional intensity (ANGER, FRUSTRATION, etc) proven in the uploaded dataset.
4. Distinguish between mild dissatisfaction and critical frustration to prioritize your strategy.

CURRENT FOCUS INSIGHT:
Question: {focus_q}
Current Proposed Solution Framework: {solution_escaped}

RAW COMPANY HISTORICAL DATA (Retrieved specifically for the user's latest query via RAG):
{docs_str_escaped}

RECENT CHAT HISTORY:
{history_str_escaped}
"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{query}")
        ])
        
        # Connecting to Groq Llama3 Model
        llm = ChatGroq(
            model=config.GROQ_STRATEGIST_MODEL,
            temperature=0.8
        )
        
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": request.query})
        
        return {"response": result}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/suggested-questions")
async def get_suggested_questions():
    try:
        # 1. Get a representative sample of reviews (mix of good and bad)
        recent_reviews = memory_bank.collection.get(limit=25, include=["documents"])
        
        if not recent_reviews.get("documents"):
            return {"questions": [
                "How can we improve our general customer experience?",
                "What are the top complaints we should address first?",
                "How do we compare to our top competitors?"
            ]}

        context_str = "\n".join([f"- {r}" for r in recent_reviews["documents"]])

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an Elite Executive Strategist. Based on the provided customer reviews, "
             "identify 3 deep, disruptive, and highly specific strategic questions that a "
             "business owner should ask an AI advisor to improve their company.\n\n"
             "Rules:\n"
             "- Questions must be derived from the specific themes in the reviews.\n"
             "- Focus on long-term growth and systemic improvements.\n"
             "- Avoid generic business advice.\n"
             "- Return a JSON array of 3 strings. No extra text."),
            
            ("user", "Reviews:\n{context}")
        ])

        llm = ChatGroq(model=config.GROQ_STRATEGIST_MODEL, temperature=0.7)
        chain = prompt | llm | StrOutputParser()
        result_str = chain.invoke({"context": context_str})

        # Regex extract JSON array
        import re
        match = re.search(r'\[.*\]', result_str, re.DOTALL)
        if match:
            questions = json.loads(match.group(0))
        else:
            # Fallback defaults if logic fails
            questions = [
                "How can we systematically eliminate the recurring wait-time issues?",
                "What disruptive service model could set us apart from local competitors?",
                "How should we realign our staff training to focus on high-impact interactions?"
            ]

        return {"questions": questions}

    except Exception as e:
        # Return friendly defaults if LLM fails
        return {"questions": [
            "How can we leverage our strengths to disrupt the market?",
            "What are the hidden systemic issues we are overlooking?",
            "What should be our priority for the next quarter?"
        ]}