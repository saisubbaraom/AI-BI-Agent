import pandas as pd

path = r"C:\Users\saisu\OneDrive\Desktop\AI-Powered Business Intelligence Agent\Mobile Sales Data.xlsx"
df = pd.read_excel(path)
print("File loaded successfully!")
print("Columns:")
print(df.columns.tolist())
print("\nShape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nInfo:")
df.info()
