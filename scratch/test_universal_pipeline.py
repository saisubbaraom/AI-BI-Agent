import os
import pandas as pd
import numpy as np
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from analytics.relationship_discovery import discover_relationships, calculate_join_stats, merge_datasets

def test_dataset_intelligence_ev():
    print("\n--- Test 1: Dataset Intelligence Agent on EV Vehicles Dataset ---")
    
    # Create sample EV Vehicles Dataset with PII and ID
    df_ev = pd.DataFrame({
        "vin_number": [f"1FTFW1ED{i}FA{10000+i}" for i in range(100)],
        "model_year": np.random.choice([2021, 2022, 2023, 2024], size=100),
        "make": np.random.choice(["Tesla", "Ford", "Chevrolet", "Nissan", "Hyundai"], size=100),
        "model": np.random.choice(["Model 3", "F-150 Lightning", "Bolt EV", "Leaf", "Ioniq 5"], size=100),
        "electric_range": np.random.randint(150, 350, size=100),
        "ev_type": np.random.choice(["BEV", "PHEV"], size=100, p=[0.8, 0.2]),
        "owner_name": [f"Owner {i}" for i in range(100)],
        "owner_phone": [f"555-01{i:02d}" for i in range(100)],
        "registration_zip": np.random.randint(90001, 98101, size=100)
    })
    
    intel = DatasetIntelligenceAgent()
    
    # Detect PII and IDs
    res_pii = intel.detect_pii_and_ids(df_ev)
    print("Detected PII columns:", res_pii["pii"])
    print("Detected ID columns:", res_pii["ids"])
    
    assert "owner_name" in res_pii["pii"], "owner_name should be detected as PII"
    assert "owner_phone" in res_pii["pii"], "owner_phone should be detected as PII"
    assert "vin_number" in res_pii["ids"], "vin_number should be detected as ID"
    
    # Run intelligence agent profiling (triggers LLM to infer EV theme and suggest KPIs)
    print("Running Dataset Intelligence profiling (calling LLM)...")
    res = intel.run(df_ev)
    
    print("Inferred Theme:", res["theme"])
    print("Inferred Domain:", res["domain"])
    print("Description:", res["description"])
    
    print("\nCalculated Custom KPIs:")
    for k, v in res["kpis"].items():
        print(f"- {v['label']} ({k}): {v['value']} (Format: {v['format']})")
        
    print("\nSuggested Visualizations:")
    for chart in res["suggested_charts"]:
        print(f"- {chart.get('title')}: type={chart.get('type')}, x={chart.get('x_col')}, y={chart.get('y_col')}")

    assert len(res["kpis"]) > 0, "Should have calculated dynamic KPIs"
    assert len(res["suggested_charts"]) > 0, "Should have generated suggested charts"
    print("Test 1 Passed!")

def test_relationship_discovery():
    print("\n--- Test 2: Relationship Discovery & Merge Planner ---")
    
    # Create mock customer and orders datasets
    df_customers = pd.DataFrame({
        "customer_id": [f"CUST-{i:03d}" for i in range(1, 101)],
        "customer_name": [f"Customer Name {i}" for i in range(1, 101)],
        "email_address": [f"customer{i}@example.com" for i in range(1, 101)],
        "signup_date": pd.date_range(start="2026-01-01", periods=100)
    })
    
    df_orders = pd.DataFrame({
        "order_id": [f"ORD-{i:04d}" for i in range(1, 201)],
        "customer_id": [f"CUST-{np.random.randint(1, 120):03d}" for _ in range(200)], # some overlap but also missing customers
        "product_id": [f"PROD-{np.random.randint(1, 10):02d}" for _ in range(200)],
        "amount": np.random.uniform(10.0, 500.0, size=200)
    })
    
    datasets = {
        "customers.csv": df_customers,
        "orders.csv": df_orders
    }
    
    # Discover relationships
    relationships = discover_relationships(datasets)
    print(f"Discovered {len(relationships)} relationships:")
    for rel in relationships:
        print(f"- {rel['source_file']}.{rel['source_col']} <-> {rel['target_file']}.{rel['target_col']} (Coverage: {rel['matching_pct']}%, Type: {rel['join_type']})")
        
    assert len(relationships) > 0, "Should discover customer_id relationship"
    best_rel = relationships[0]
    assert best_rel["source_col"] == "customer_id", "Should match customer_id"
    
    # Calculate detailed join stats
    stats = calculate_join_stats(
        df_customers, 
        df_orders, 
        best_rel["source_col"], 
        best_rel["target_col"], 
        "left"
    )
    print("\nJoin Stats (Left Join Customers -> Orders):")
    for k, v in stats.items():
        if isinstance(v, str):
            print(f"- {k}: {v.encode('ascii', errors='replace').decode('ascii')}")
        else:
            print(f"- {k}: {v}")
        
    # Perform actual merge
    merged_df = merge_datasets(
        df_customers, 
        df_orders, 
        best_rel["source_col"], 
        best_rel["target_col"], 
        "left"
    )
    print(f"\nMerged DataFrame Shape: {merged_df.shape}")
    print("Merged columns:", list(merged_df.columns))
    
    assert merged_df.shape[0] >= len(df_customers), "Merged row count should be at least customer size for left join"
    print("Test 2 Passed!")

if __name__ == "__main__":
    test_dataset_intelligence_ev()
    test_relationship_discovery()
