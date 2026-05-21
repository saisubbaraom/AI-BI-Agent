import pandas as pd
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "data")

for f in os.listdir(data_dir):
    path = os.path.join(data_dir, f)
    if os.path.isfile(path) and f.endswith(('.csv', '.xlsx', '.parquet')):
        print(f"=== {f} ===")
        if f.endswith('.csv'):
            df = pd.read_csv(path)
        elif f.endswith('.xlsx'):
            df = pd.read_excel(path)
        elif f.endswith('.parquet'):
            df = pd.read_parquet(path)
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)
        print("Sample Row:", df.iloc[0].to_dict())
        print()
