import os
import pandas as pd
from agents.master_agent import MasterAgent
from reports.pdf_generator import generate_pdf_report

def main():
    print("=== Testing PDF Generation ===")
    
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
    
    print("Attempting to generate PDF...")
    try:
        pdf_data = generate_pdf_report(bundle)
        print("Success! PDF generated. Length of bytes:", len(pdf_data))
    except Exception as e:
        print("Failed with error:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
