import os
import pandas as pd
from agents.master_agent import MasterAgent

def main():
    print("=== Business Analyst AI Agent Pipeline Verification ===")
    
    # Path to generated sample sales data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sales_path = os.path.join(base_dir, "data", "sales_data.csv")
    
    if not os.path.exists(sales_path):
        print(f"ERROR: Sales sample dataset not found at {sales_path}. Generating now...")
        from generate_sample_data import generate_datasets
        generate_datasets()
        
    print(f"Loading test dataset from: {sales_path}")
    df = pd.read_csv(sales_path)
    print(f"Dataset shape: {df.shape}")
    
    print("\nInitializing MasterAgent and starting pipeline run...")
    master = MasterAgent()
    
    try:
        # Run with default cleaning options
        bundle = master.run(df, clean_data=True)
        
        print("\n--- Pipeline Completed Successfully! ---")
        print(f"Domain Detected: {bundle['domain']}")
        print(f"Columns Mapped: {bundle['col_map']}")
        print("\nCalculated KPIs:")
        for k, v in bundle["kpis"].items():
            print(f"- {v['label']}: {v['value']} ({v['format']})")
            
        print("\nGenerated Charts:")
        for name, details in bundle["charts"].items():
            print(f"- {name}: {details['title']} ({details['type']})")
            
        print("\nRecommendations Count:")
        print(f"Total Recommendations: {len(bundle['recommendations'])}")
        for r in bundle['recommendations'][:2]:
            print(f"- [{r.get('priority')}] {r.get('title')}: {r.get('description')[:60]}...")
            
        print("\nQuality Profile Sample (null percent):")
        for col, q in list(bundle["quality_report"].items())[:3]:
            print(f"- {col}: Missing {q['null_percent']}%")
            
        print("\nAll systems operational! The multi-agent pipeline compiles and runs successfully.")
        
    except Exception as e:
        print(f"\n[FAIL] Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
