from langgraph.graph import StateGraph, END
from state import GraphState
from agents import triage_agent, prioritization_agent, synthesizer_agent
from forge import forge_agent

# Initialize the workflow graph
workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("triage", triage_agent)
workflow.add_node("prioritization", prioritization_agent)
workflow.add_node("synthesizer", synthesizer_agent)
workflow.add_node("forge", forge_agent)

# Define the edges (Flow) -> Triage -> Prioritization -> Synthesizer -> Forge -> END
workflow.set_entry_point("triage")
workflow.add_edge("triage", "prioritization")
workflow.add_edge("prioritization", "synthesizer")
workflow.add_edge("synthesizer", "forge")
workflow.add_edge("forge", END)

# Compile the graph
feedback_pipeline = workflow.compile()
