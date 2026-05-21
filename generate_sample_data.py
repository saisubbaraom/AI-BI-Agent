import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_datasets():
    # Create target directory
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(target_dir, exist_ok=True)
    
    np.random.seed(42)
    start_date = datetime(2026, 1, 1)
    
    # ----------------------------------------------------
    # 1. SALES DATASET (CSV)
    # ----------------------------------------------------
    sales_rows = 500
    dates = [start_date + timedelta(days=int(np.random.randint(0, 90))) for _ in range(sales_rows)]
    products = np.random.choice(["Laptops", "Smartphones", "Accessories", "Smart Watches", "Monitors"], size=sales_rows, p=[0.15, 0.25, 0.35, 0.15, 0.10])
    
    base_prices = {"Laptops": 1200.0, "Smartphones": 800.0, "Accessories": 45.0, "Smart Watches": 250.0, "Monitors": 350.0}
    cogs_pct = {"Laptops": 0.70, "Smartphones": 0.65, "Accessories": 0.35, "Smart Watches": 0.50, "Monitors": 0.60}
    
    quantities = np.random.randint(1, 5, size=sales_rows)
    revenues = [base_prices[prod] * qty for prod, qty in zip(products, quantities)]
    costs = [rev * cogs_pct[prod] for prod, rev in zip(products, revenues)]
    
    orders = [f"ORD-{1000 + i}" for i in range(sales_rows)]
    customers = [f"CUST-{100 + np.random.randint(1, 150)}" for _ in range(sales_rows)]
    
    # Customer Demographics
    ages = np.random.randint(18, 70, size=sales_rows)
    genders = np.random.choice(["Male", "Female", "Non-binary"], size=sales_rows, p=[0.48, 0.48, 0.04])
    locations = np.random.choice(["New York", "California", "Texas", "Florida", "Washington", "Illinois"], size=sales_rows)
    
    df_sales = pd.DataFrame({
        "order_id": orders,
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "product_category": products,
        "quantity": quantities,
        "revenue": revenues,
        "cost": costs,
        "customer_id": customers,
        "age": ages,
        "gender": genders,
        "location": locations
    })
    
    sales_path = os.path.join(target_dir, "sales_data.csv")
    df_sales.to_csv(sales_path, index=False)
    print(f"Generated Sales data: {sales_path}")

    # ----------------------------------------------------
    # 2. MARKETING CAMPAIGN DATASET (XLSX)
    # ----------------------------------------------------
    mktg_rows = 150
    m_dates = [start_date + timedelta(days=int(np.random.randint(0, 90))) for _ in range(mktg_rows)]
    campaigns = np.random.choice(["Spring Promo", "Search Ads Core", "Social Media Branding", "Retargeting Loop", "Affiliate Blast"], size=mktg_rows)
    
    # Generate proportional spend and impressions
    base_spends = {"Spring Promo": 500, "Search Ads Core": 1200, "Social Media Branding": 800, "Retargeting Loop": 400, "Affiliate Blast": 600}
    spends = [base_spends[camp] * np.random.uniform(0.8, 1.2) for camp in campaigns]
    
    impressions = [int(spend * np.random.uniform(80, 120)) for spend in spends]
    clicks = [int(imp * np.random.uniform(0.015, 0.04)) for imp in impressions] # 1.5% to 4% CTR
    conversions = [int(click * np.random.uniform(0.02, 0.08)) for click in clicks] # 2% to 8% CVR
    
    # Average purchase value of $120
    revenue_gen = [conv * 120.0 * np.random.uniform(0.9, 1.1) for conv in conversions]
    
    df_mktg = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in m_dates],
        "campaign_name": campaigns,
        "spend": spends,
        "impressions": impressions,
        "clicks": clicks,
        "conversions": conversions,
        "revenue_generated": revenue_gen
    })
    
    mktg_path = os.path.join(target_dir, "marketing_data.xlsx")
    df_mktg.to_excel(mktg_path, index=False)
    print(f"Generated Marketing data: {mktg_path}")

    # ----------------------------------------------------
    # 3. HR DATASET (PARQUET)
    # ----------------------------------------------------
    hr_rows = 300
    depts = np.random.choice(["Engineering", "Sales", "Marketing", "HR", "Finance", "Product"], size=hr_rows, p=[0.30, 0.25, 0.15, 0.08, 0.07, 0.15])
    
    base_salaries = {"Engineering": 105000, "Sales": 75000, "Marketing": 70000, "HR": 65000, "Finance": 85000, "Product": 95000}
    salaries = [int(base_salaries[d] * np.random.uniform(0.85, 1.15)) for d in depts]
    
    perf_scores = np.random.choice([1, 2, 3, 4, 5], size=hr_rows, p=[0.05, 0.15, 0.55, 0.20, 0.05])
    
    # Attrition probability correlation with performance score and salaries
    attrition = []
    for sal, perf, dept in zip(salaries, perf_scores, depts):
        prob = 0.15
        if perf <= 2:
            prob += 0.20
        if sal < base_salaries[dept]:
            prob += 0.10
        attrition.append(np.random.choice(["Yes", "No"], p=[prob, 1 - prob]))
        
    emp_ids = [f"EMP-{1000 + i}" for i in range(hr_rows)]
    h_genders = np.random.choice(["Male", "Female"], size=hr_rows)
    h_ages = np.random.randint(22, 60, size=hr_rows)
    
    df_hr = pd.DataFrame({
        "employee_id": emp_ids,
        "department": depts,
        "salary": salaries,
        "performance_score": perf_scores,
        "attrition": attrition,
        "gender": h_genders,
        "age": h_ages
    })
    
    hr_path = os.path.join(target_dir, "hr_data.parquet")
    df_hr.to_parquet(hr_path, index=False)
    print(f"Generated HR data: {hr_path}")

if __name__ == "__main__":
    generate_datasets()
