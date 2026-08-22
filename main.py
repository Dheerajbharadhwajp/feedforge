import config
from memory import memory_bank
from graph import feedback_pipeline

def main():
    print("=== Welcome to FeedbackForge ===")
    
    # 1. Initialize Memory Bank (Layer 1)
    memory_bank.load_data()
    
    # 2. Pick a few test reviews to run through the Multi-Agent Pipeline
    test_reviews = [
        {
            "review_id": "TEST_001",
            "review_text": "I ordered through the app and when I got to the store they had no record of it. The barista was incredibly rude to me about it and my card was charged $15!",
            "metadata": {"rating": 1, "review_length": 145}
        },
        {
            "review_id": "TEST_002",
            "review_text": "Me and my Special needs son need to wait at a Starbucks for an Access Van. The store has only teeny tiny tables, where two people can not sit at, as the table is far too small. The staff, told me Starbucks is making it extremely uncomfortable for a family to eat inside",
            "metadata": {"rating": 2, "review_length": 257}
        }
    ]
    
    print("\\nStarting Multi-Agent Orchestrator Pipeline (Layer 2 & 3)...\\n")
    
    for initial_state in test_reviews:
        print(f"\\n>>>> PROCESSING TARGET: {initial_state['review_id']} <<<<")
        print(f"Review: '{initial_state['review_text']}'\\n")
        
        # Execute the LangGraph pipeline
        result_state = feedback_pipeline.invoke(initial_state)
        
        print("\\n--- PIPELINE RESULTS ---")
        print(f"Category: {result_state['category']}")
        print(f"Urgency Score: {result_state['urgency_score']}")
        print(f"Identified Root Cause:\\n{result_state['root_cause']}")
        print(f"\\n--- FORGE FINAL OUTPUT ({result_state['category']}) ---")
        print(result_state['final_output'])
        print("="*60)

if __name__ == "__main__":
    main()
