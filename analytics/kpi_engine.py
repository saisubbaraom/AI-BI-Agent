import pandas as pd
import numpy as np

def detect_domain_and_columns(df: pd.DataFrame) -> tuple[str, dict]:
    """
    Analyzes the column names of the dataset to detect the business domain
    and maps columns to standardized business concepts.
    Returns: (domain_name, column_mapping)
    """
    cols = [col.lower() for col in df.columns]
    col_map = {orig: orig.lower() for orig in df.columns}
    
    # Define keywords for domain detection
    sales_keywords = {
        "revenue": ["revenue", "sales", "amount", "total_price", "grand_total", "price", "subtotal"],
        "order_id": ["order_id", "transaction_id", "invoice_no", "order_number", "receipt_id"],
        "quantity": ["quantity", "qty", "units_sold", "volume", "count"],
        "cost": ["cost", "cogs", "expense_amount", "purchase_price"],
        "customer_id": ["customer_id", "user_id", "email", "member_id", "client_id"],
        "date": ["date", "timestamp", "order_date", "created_at", "invoice_date"]
    }
    
    marketing_keywords = {
        "spend": ["spend", "cost", "budget", "ad_spend", "marketing_spend"],
        "impressions": ["impressions", "views", "reach"],
        "clicks": ["clicks", "taps"],
        "conversions": ["conversions", "signups", "purchases", "leads", "actions"],
        "revenue": ["revenue", "sales", "conversion_value", "revenue_generated"],
        "campaign": ["campaign", "campaign_name", "ad_group", "source"]
    }
    
    hr_keywords = {
        "attrition": ["attrition", "left", "terminated", "status", "exit_date"],
        "salary": ["salary", "pay", "income", "compensation", "wage"],
        "employee_id": ["employee_id", "emp_id", "staff_id", "serial_number"],
        "performance": ["performance", "rating", "evaluation", "score", "review"],
        "department": ["department", "dept", "team", "division"],
        "date": ["hire_date", "joined_date", "start_date", "date_of_joining"]
    }
    
    finance_keywords = {
        "revenue": ["revenue", "sales", "income", "gross_profit"],
        "expense": ["expense", "expenses", "spend", "cost", "outflow", "payments"],
        "profit": ["profit", "net_profit", "earnings", "net_income"],
        "assets": ["assets", "cash", "inventory"],
        "liabilities": ["liabilities", "debt", "payables"]
    }

    # Helper to calculate matches
    def score_domain(keywords_dict):
        score = 0
        matches = {}
        for concept, kw_list in keywords_dict.items():
            matched_col = None
            for c in df.columns:
                c_clean = c.lower().replace("_", "").replace(" ", "")
                for kw in kw_list:
                    kw_clean = kw.replace("_", "").replace(" ", "")
                    # Match exact or substring
                    if kw_clean in c_clean or c_clean in kw_clean:
                        matched_col = c
                        break
                if matched_col:
                    break
            if matched_col:
                score += 2  # high weight for keyword matches
                matches[concept] = matched_col
        return score, matches

    sales_score, sales_matches = score_domain(sales_keywords)
    mktg_score, mktg_matches = score_domain(marketing_keywords)
    hr_score, hr_matches = score_domain(hr_keywords)
    fin_score, fin_matches = score_domain(finance_keywords)
    
    scores = {
        "Sales": (sales_score, sales_matches),
        "Marketing": (mktg_score, mktg_matches),
        "HR": (hr_score, hr_matches),
        "Finance": (fin_score, fin_matches)
    }
    
    best_domain = "Generic"
    best_score = 1  # Threshold score to classify as specialized
    mapped_columns = {}
    
    for domain, (score, matches) in scores.items():
        if score > best_score:
            best_score = score
            best_domain = domain
            mapped_columns = matches
            
    # Try to auto-detect any date column even in generic domains
    if "date" not in mapped_columns:
        for c in df.columns:
            c_lower = c.lower()
            if any(k in c_lower for k in ["date", "time", "created_at", "timestamp", "year", "month"]):
                mapped_columns["date"] = c
                break
                
    return best_domain, mapped_columns

def safe_divide(numerator, denominator, fallback=0.0):
    """Safely divide to prevent ZeroDivisionError."""
    try:
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return fallback
        return float(numerator / denominator)
    except:
        return fallback

def compute_kpis(df: pd.DataFrame, domain: str, col_map: dict) -> dict:
    """
    Computes business-level KPIs based on detected domain and mapped columns.
    """
    kpis = {}
    
    # Identify standard date column if any
    date_col = col_map.get("date")
    dates_valid = False
    if date_col:
        try:
            df_dates = pd.to_datetime(df[date_col], errors='coerce')
            dates_valid = not df_dates.isnull().all()
        except:
            dates_valid = False

    if domain == "Sales":
        rev_col = col_map.get("revenue")
        ord_col = col_map.get("order_id")
        qty_col = col_map.get("quantity")
        cost_col = col_map.get("cost")
        cust_col = col_map.get("customer_id")
        
        # 1. Total Revenue
        total_rev = df[rev_col].sum() if rev_col else 0.0
        kpis["total_revenue"] = {"label": "Total Revenue", "value": total_rev, "format": "currency"}
        
        # 2. Total Orders
        total_orders = df[ord_col].nunique() if ord_col else len(df)
        kpis["total_orders"] = {"label": "Total Orders", "value": total_orders, "format": "integer"}
        
        # 3. Average Order Value (AOV)
        aov = safe_divide(total_rev, total_orders)
        kpis["average_order_value"] = {"label": "Average Order Value", "value": aov, "format": "currency"}
        
        # 4. Total Units Sold
        total_qty = df[qty_col].sum() if qty_col else 0.0
        if qty_col:
            kpis["total_units_sold"] = {"label": "Total Units Sold", "value": total_qty, "format": "integer"}
            
        # 5. Profit Margin
        if cost_col and rev_col:
            total_cost = df[cost_col].sum()
            profit_margin = safe_divide(total_rev - total_cost, total_rev)
            kpis["profit_margin"] = {"label": "Gross Profit Margin", "value": profit_margin, "format": "percent"}
            
        # 6. Customer Lifetime Value (CLV - Average Revenue Per Customer)
        if cust_col:
            unique_custs = df[cust_col].nunique()
            clv = safe_divide(total_rev, unique_custs)
            kpis["clv"] = {"label": "Avg. Value per Customer", "value": clv, "format": "currency"}
            kpis["unique_customers"] = {"label": "Unique Customers", "value": unique_custs, "format": "integer"}

    elif domain == "Marketing":
        spend_col = col_map.get("spend")
        imp_col = col_map.get("impressions")
        click_col = col_map.get("clicks")
        conv_col = col_map.get("conversions")
        rev_col = col_map.get("revenue")
        
        total_spend = df[spend_col].sum() if spend_col else 0.0
        kpis["total_spend"] = {"label": "Total Ad Spend", "value": total_spend, "format": "currency"}
        
        total_convs = df[conv_col].sum() if conv_col else 0.0
        kpis["total_conversions"] = {"label": "Total Conversions", "value": total_convs, "format": "integer"}
        
        # CPA
        cpa = safe_divide(total_spend, total_convs)
        kpis["cpa"] = {"label": "Cost Per Acquisition (CPA)", "value": cpa, "format": "currency"}
        
        # CTR
        if click_col and imp_col:
            total_clicks = df[click_col].sum()
            total_imps = df[imp_col].sum()
            ctr = safe_divide(total_clicks, total_imps)
            kpis["ctr"] = {"label": "Avg. Click-Through Rate (CTR)", "value": ctr, "format": "percent"}
            
            cpc = safe_divide(total_spend, total_clicks)
            kpis["cpc"] = {"label": "Average Cost Per Click (CPC)", "value": cpc, "format": "currency"}
            
        # Conversion Rate
        if click_col and conv_col:
            total_clicks = df[click_col].sum() if click_col else 1.0
            cvr = safe_divide(total_convs, total_clicks)
            kpis["conversion_rate"] = {"label": "Conversion Rate (CVR)", "value": cvr, "format": "percent"}
            
        # ROAS
        if rev_col and spend_col:
            total_rev = df[rev_col].sum()
            roas = safe_divide(total_rev, total_spend)
            kpis["roas"] = {"label": "Return on Ad Spend (ROAS)", "value": roas, "format": "number"}

    elif domain == "HR":
        attr_col = col_map.get("attrition")
        salary_col = col_map.get("salary")
        emp_col = col_map.get("employee_id")
        perf_col = col_map.get("performance")
        
        headcount = df[emp_col].nunique() if emp_col else len(df)
        kpis["headcount"] = {"label": "Active Headcount", "value": headcount, "format": "integer"}
        
        # Attrition Rate
        if attr_col:
            # Check if attrition is string (Yes/No) or numeric (1/0)
            attr_series = df[attr_col].dropna()
            if attr_series.dtype == object or attr_series.dtype == bool:
                attr_count = attr_series.astype(str).str.lower().isin(["yes", "1", "true", "left", "terminated"]).sum()
            else:
                attr_count = (attr_series == 1).sum()
                
            attr_rate = safe_divide(attr_count, headcount)
            kpis["attrition_rate"] = {"label": "Attrition Rate", "value": attr_rate, "format": "percent"}
            kpis["attrition_count"] = {"label": "Employees Left", "value": attr_count, "format": "integer"}
            
        # Average Salary
        if salary_col:
            avg_salary = df[salary_col].mean()
            kpis["average_salary"] = {"label": "Average Salary", "value": avg_salary, "format": "currency"}
            
        # Average Performance
        if perf_col:
            avg_perf = df[perf_col].mean()
            kpis["average_performance"] = {"label": "Avg. Performance Score", "value": avg_perf, "format": "number"}

    elif domain == "Finance":
        rev_col = col_map.get("revenue")
        exp_col = col_map.get("expense")
        prof_col = col_map.get("profit")
        asset_col = col_map.get("assets")
        liab_col = col_map.get("liabilities")
        
        total_rev = df[rev_col].sum() if rev_col else 0.0
        kpis["total_revenue"] = {"label": "Revenue", "value": total_rev, "format": "currency"}
        
        total_exp = df[exp_col].sum() if exp_col else 0.0
        kpis["total_expenses"] = {"label": "Expenses", "value": total_exp, "format": "currency"}
        
        # Net Profit
        if prof_col:
            net_profit = df[prof_col].sum()
        else:
            net_profit = total_rev - total_exp
        kpis["net_profit"] = {"label": "Net Profit", "value": net_profit, "format": "currency"}
        
        # Profit Margin
        if total_rev > 0:
            profit_margin = safe_divide(net_profit, total_rev)
            kpis["net_profit_margin"] = {"label": "Net Profit Margin", "value": profit_margin, "format": "percent"}
            
        # ROI
        if total_exp > 0:
            roi = safe_divide(net_profit, total_exp)
            kpis["roi"] = {"label": "Return on Investment (ROI)", "value": roi, "format": "percent"}
            
        # Asset to Liability Ratio
        if asset_col and liab_col:
            total_assets = df[asset_col].sum()
            total_liab = df[liab_col].sum()
            al_ratio = safe_divide(total_assets, total_liab)
            kpis["asset_liability_ratio"] = {"label": "Asset/Liability Ratio", "value": al_ratio, "format": "number"}

    # Generic or Fallback metrics
    if not kpis:
        # Find numeric columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude common ID columns
        num_cols = [c for c in num_cols if not c.lower().endswith("id")]
        
        if num_cols:
            # Choose up to 4 columns to compute generic stats
            for c in num_cols[:4]:
                kpis[f"sum_{c}"] = {"label": f"Total {c.replace('_', ' ').title()}", "value": df[c].sum(), "format": "number"}
                kpis[f"avg_{c}"] = {"label": f"Avg. {c.replace('_', ' ').title()}", "value": df[c].mean(), "format": "number"}
        else:
            # Fallback when there are no numeric columns
            kpis["row_count"] = {"label": "Total Records", "value": len(df), "format": "integer"}
            kpis["col_count"] = {"label": "Total Features", "value": len(df.columns), "format": "integer"}
            
    return kpis
