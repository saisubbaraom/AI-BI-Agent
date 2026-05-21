import pandas as pd
import numpy as np
from utils.data_cleaner import detect_outliers_iqr

def get_dataset_overview(df: pd.DataFrame) -> dict:
    """
    Calculates high-level stats about the dataset.
    """
    # Calculate memory usage in MB
    memory_bytes = df.memory_usage(deep=True).sum()
    memory_mb = memory_bytes / (1024 * 1024)
    
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "duplicate_rows": df.duplicated().sum(),
        "memory_usage_mb": round(memory_mb, 2),
        "column_names": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
    }

def get_data_quality_report(df: pd.DataFrame) -> dict:
    """
    Analyzes missing values and outliers per column.
    """
    quality = {}
    
    for col in df.columns:
        null_count = df[col].isnull().sum()
        null_percent = (null_count / len(df)) * 100 if len(df) > 0 else 0
        
        col_type = str(df[col].dtype)
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        outliers_count = 0
        if is_numeric and not col.lower().endswith("id"):
            outliers = detect_outliers_iqr(df, col)
            outliers_count = int(outliers.sum())
            
        quality[col] = {
            "dtype": col_type,
            "null_count": int(null_count),
            "null_percent": round(null_percent, 2),
            "outliers_count": outliers_count,
            "is_numeric": is_numeric
        }
        
    return quality

def detect_demographic_columns(df: pd.DataFrame) -> dict:
    """
    Scan column names to automatically detect demographic categories:
    Age, Gender, Location, Income, Occupation
    """
    demographics = {
        "age": None,
        "gender": None,
        "location": None,
        "income": None,
        "occupation": None
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        if not demographics["age"] and col_lower in ["age", "customer_age", "dob", "birth_year"]:
            demographics["age"] = col
        elif not demographics["gender"] and col_lower in ["gender", "sex", "customer_gender"]:
            demographics["gender"] = col
        elif not demographics["location"] and col_lower in ["location", "city", "country", "state", "region", "zipcode", "postcode", "address"]:
            demographics["location"] = col
        elif not demographics["income"] and col_lower in ["income", "annual_income", "salary", "household_income", "earnings"]:
            demographics["income"] = col
        elif not demographics["occupation"] and col_lower in ["occupation", "job", "profession", "industry", "work_type"]:
            demographics["occupation"] = col
            
    return demographics

def analyze_demographics(df: pd.DataFrame) -> dict:
    """
    Generates summary analysis for detected demographic columns.
    """
    cols = detect_demographic_columns(df)
    analysis = {}
    
    # 1. Age analysis
    age_col = cols["age"]
    if age_col and pd.api.types.is_numeric_dtype(df[age_col]):
        ages = df[age_col].dropna()
        if not ages.empty:
            # Create standard age bins
            bins = [0, 18, 25, 35, 45, 60, 120]
            labels = ["Under 18", "18-24", "25-34", "35-44", "45-59", "60+"]
            age_groups = pd.cut(ages, bins=bins, labels=labels, right=False)
            dist = age_groups.value_counts(normalize=True).to_dict()
            
            analysis["age"] = {
                "column": age_col,
                "mean": round(float(ages.mean()), 1),
                "median": float(ages.median()),
                "min": float(ages.min()),
                "max": float(ages.max()),
                "distribution": {k: round(v * 100, 2) for k, v in dist.items()}
            }
            
    # 2. Gender analysis
    gender_col = cols["gender"]
    if gender_col:
        genders = df[gender_col].dropna()
        if not genders.empty:
            dist = genders.value_counts(normalize=True).to_dict()
            analysis["gender"] = {
                "column": gender_col,
                "distribution": {str(k): round(v * 100, 2) for k, v in dist.items()}
            }
            
    # 3. Location analysis
    loc_col = cols["location"]
    if loc_col:
        locations = df[loc_col].dropna()
        if not locations.empty:
            # Keep top 10 and group the rest
            top_counts = locations.value_counts()
            top_n = top_counts.head(10).to_dict()
            total = len(locations)
            
            dist = {str(k): round((v / total) * 100, 2) for k, v in top_n.items()}
            if len(top_counts) > 10:
                others_sum = top_counts.iloc[10:].sum()
                dist["Other"] = round((others_sum / total) * 100, 2)
                
            analysis["location"] = {
                "column": loc_col,
                "unique_count": len(top_counts),
                "distribution": dist
            }
            
    # 4. Income analysis
    income_col = cols["income"]
    if income_col and pd.api.types.is_numeric_dtype(df[income_col]):
        incomes = df[income_col].dropna()
        if not incomes.empty:
            # Create income bins
            min_inc, max_inc = incomes.min(), incomes.max()
            # If values seem to be annual (e.g. > 1000)
            if max_inc > 1000:
                bins = [0, 30000, 60000, 90000, 120000, 200000, np.inf]
                labels = ["< $30K", "$30K-$60K", "$60K-$90K", "$90K-$120K", "$120K-$200K", "$200K+"]
            else:
                bins = [0, 2, 4, 6, 8, 10, np.inf]
                labels = ["Very Low", "Low", "Medium", "High", "Very High", "Ultra"]
                
            income_groups = pd.cut(incomes, bins=bins, labels=labels, right=False)
            dist = income_groups.value_counts(normalize=True).to_dict()
            
            analysis["income"] = {
                "column": income_col,
                "mean": round(float(incomes.mean()), 2),
                "median": float(incomes.median()),
                "distribution": {k: round(v * 100, 2) for k, v in dist.items()}
            }
            
    # 5. Occupation analysis
    occ_col = cols["occupation"]
    if occ_col:
        occs = df[occ_col].dropna()
        if not occs.empty:
            top_counts = occs.value_counts()
            top_n = top_counts.head(5).to_dict()
            total = len(occs)
            
            dist = {str(k): round((v / total) * 100, 2) for k, v in top_n.items()}
            if len(top_counts) > 5:
                others_sum = top_counts.iloc[5:].sum()
                dist["Other"] = round((others_sum / total) * 100, 2)
                
            analysis["occupation"] = {
                "column": occ_col,
                "unique_count": len(top_counts),
                "distribution": dist
            }
            
    return analysis
