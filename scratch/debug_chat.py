import os
import pandas as pd
from utils.config import get_llm
from agents.master_agent import MasterAgent

# Load data
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sales_path = os.path.join(base_dir, "data", "sales_data.csv")
df = pd.read_csv(sales_path)

# Run MasterAgent to get doc store and agent
master = MasterAgent()
bundle = master.run(df, clean_data=True)

chat_agent = bundle["chat_agent"]
query_engine = chat_agent.query_engine

questions = [
    "which brand sales are high",
    "means which made profits high"
]

history = []

for q in questions:
    print(f"\n======================================")
    print(f"QUESTION: '{q}'")
    print(f"======================================")
    
    # 1. Check query plan with history
    plan = query_engine.generate_query_plan(q, chat_history=history)
    print("\n--- QUERY PLAN ---")
    import json
    print(json.dumps(plan, indent=2))
    
    # 2. Check query execution
    res = query_engine.execute_query_plan(plan)
    print("\n--- QUERY OUTPUT ---")
    print(res)

    # 3. Check ChatAgent Run response
    response_text = chat_agent.run(q, history)
    print("\n--- CHATBOT AGENT RESPONSE ---")
    safe_response = response_text.encode('ascii', errors='replace').decode('ascii')
    print(safe_response)

    # Append to simulated history
    history.append({"role": "user", "content": q})
    history.append({"role": "assistant", "content": response_text})
