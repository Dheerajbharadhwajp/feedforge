import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

llm = ChatGroq(
    model=config.GROQ_FAST_MODEL,
    temperature=0.2
)

def forge_agent(state):
    """Generates a Product Requirement Document (PRD) or a Customer Support Email Template."""
    print(f"[{state['review_id']}] The Forge: Generating actionable output...")
    
    urgency = state.get("urgency_score", 0)
    category = state.get("category", "Service")
    root_cause = state.get("root_cause", "")
    
    if category in ["Product", "App"]:
        output_format = "Product Requirement Document (PRD) snippet"
        prompt_sys = f"You are the Lead Product Strategist. Create a brief {output_format} addressing the root cause. Include: Problem Statement, Proposed Feature, and Acceptance Criteria."
    else:
        output_format = "Customer Support Email Template or Operational Directive"
        prompt_sys = f"You are the Operations Team Lead. Answer with an {output_format} addressing the root cause. If urgency > 3.5, frame it as a critical operational directive to staff. Otherwise, frame it as an empathetic email to the customer."
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_sys),
        ("user", f"Urgency Score: {urgency}/10\\nRoot Cause Identified: {root_cause}\\n\\nGenerate the Output:")
    ])
    
    chain = prompt | llm | StrOutputParser()
    final_output = chain.invoke({})
    
    return {"final_output": final_output}
