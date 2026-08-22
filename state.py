from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    Represents the state of our agentic workflow during LangGraph traversal.
    """
    review_id: str
    review_text: str
    metadata: Dict[str, Any]
    
    # Updated by Triage Agent
    category: Optional[str]
    
    # Updated by Prioritization Agent
    urgency_score: Optional[float]
    
    # Updated by Synthesizer Agent
    root_cause: Optional[str]
    similar_reviews: Optional[List[str]]
    
    # Updated by Forge Agent
    final_output: Optional[str]
