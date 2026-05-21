import os
import sys

# Ensure project root is in PYTHONPATH
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import pandas as pd
from agents.master_agent import MasterAgent
from reports.docx_generator import generate_docx_report

def main():
    print("=== Testing DOCX Generation ===")
    
    # Path to generated sample sales data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sales_path = os.path.join(base_dir, "data", "sales_data.csv")
    
    if not os.path.exists(sales_path):
        print(f"Generating sample dataset at {sales_path}...")
        from generate_sample_data import generate_datasets
        generate_datasets()
        
    df = pd.read_csv(sales_path)
    master = MasterAgent()
    bundle = master.run(df, clean_data=True)
    
    print("Attempting to generate DOCX...")
    try:
        docx_data = generate_docx_report(bundle)
        print("Success! DOCX generated. Length of bytes:", len(docx_data))
        
        # Write to temporary file for verification
        out_path = os.path.join(base_dir, "scratch", "test_report.docx")
        with open(out_path, "wb") as f:
            f.write(docx_data)
        print(f"DOCX report saved to {out_path}")
    except Exception as e:
        print("Failed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
