import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config 
from memory import memory_bank

# Initialize Groq LLM
llm = ChatGroq(
    model=config.GROQ_FAST_MODEL,
    temperature=0
)

# --- TRIAGE AGENT ---

def triage_agent(state):
    """Categorizes the review into [Service, Product, App, Atmosphere]."""
    print(f"[{state['review_id']}] Triage Agent: Classifying...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Triage Agent for FeedbackForge. You must categorize the given customer review into exactly ONE of the following categories: Service, Product, App, Atmosphere. Output ONLY the category name, nothing else."),
        ("user", "Review: {review_text}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    category = chain.invoke({"review_text": state["review_text"]})
    
    valid_categories = ["Service", "Product", "App", "Atmosphere"]
    category = category.strip().strip('"').strip("'")
    
    matched_cat = "Service" # fallback
    for cat in valid_categories:
        if cat.lower() in category.lower():
            matched_cat = cat
            break
            
    return {"category": matched_cat}

# --- PRIORITIZATION AGENT ---

def prioritization_agent(state):
    """Calculates an Urgency Score using the formula: (5 - Rating) + (Review_Length / 200)."""
    print(f"[{state['review_id']}] Prioritization Agent: Calculating urgency...")
    
    rating = state["metadata"].get("rating", 3)
    review_length = state["metadata"].get("review_length", len(state["review_text"]))
    
    urgency_score = (5 - rating) + (review_length / 200.0)
    
    return {"urgency_score": round(urgency_score, 2)}

# --- SYNTHESIZER AGENT ---

def synthesizer_agent(state):
    """Queries similar reviews and identifies root causes."""
    print(f"[{state['review_id']}] Synthesizer Agent: Identifying root causes...")
    
    # Query Memory for similar complaints to augment generation
    results = memory_bank.search_similar(query=state["review_text"], n_results=5)
    
    similar_texts = []
    if results and "documents" in results and len(results["documents"]) > 0:
        similar_texts = results["documents"][0]
        
    context_str = "\\n\\n".join(similar_texts)
    
    # Synthesize root cause from this cluster
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the Synthesizer Agent. You are given a primary customer review and a few similar reviews. Your goal is to extract the underlying 'Root Cause' of the problem. Keep it to 2-3 concise sentences."),
        ("user", "Primary Review:\\n{review_text}\\n\\nSimilar Reviews:\\n{context_str}\\n\\nIdentify the Root Cause:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    root_cause = chain.invoke({
        "review_text": state["review_text"],
        "context_str": context_str
    })
    
    return {"root_cause": root_cause, "similar_reviews": similar_texts}
