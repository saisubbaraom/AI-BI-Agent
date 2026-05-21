import os
import pandas as pd
import traceback
from agents.master_agent import MasterAgent

def main():
    path = r"C:\Users\saisu\OneDrive\Desktop\AI-Powered Business Intelligence Agent\Mobile Sales Data.xlsx"
    print(f"Loading user dataset from: {path}")
    
    try:
        df = pd.read_excel(path, engine="openpyxl")
        print(f"Dataset loaded. Shape: {df.shape}")
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3))
        
        print("\nInitializing MasterAgent...")
        master = MasterAgent()
        
        print("\nRunning pipeline...")
        bundle = master.run(df, clean_data=True)
        
        print("\n--- Pipeline Completed Successfully! ---")
        print(f"Domain Detected: {bundle['domain']}")
        print(f"Columns Mapped: {bundle['col_map']}")
        print("\nCalculated KPIs:")
        for k, v in bundle["kpis"].items():
            print(f"- {v['label']}: {v['value']} ({v['format']})")
        
        print("\nRecommendations Count:", len(bundle['recommendations']))
        print("\nInsights keys:", bundle['insights'].keys())
    except Exception as e:
        print("\n[FAIL] Pipeline failed with error:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
