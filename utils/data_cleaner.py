import pandas as pd
import numpy as np

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans column names to make them standard and easy to reference:
    e.g., 'Product Name' -> 'product_name', 'Sales ($)' -> 'sales'
    """
    df_cleaned = df.copy()
    new_cols = {}
    for col in df_cleaned.columns:
        # Convert to string and strip spaces
        clean_name = str(col).strip()
        # Replace spaces, hyphens, slashes with underscores
        clean_name = clean_name.replace(" ", "_").replace("-", "_").replace("/", "_").replace("\\", "_")
        # Remove special characters like parentheses, dollar signs, percent signs, etc.
        for char in ["(", ")", "$", "%", "#", "@", "!", "?", "*", "[", "]", "{", "}"]:
            clean_name = clean_name.replace(char, "")
        # Remove multiple consecutive underscores
        while "__" in clean_name:
            clean_name = clean_name.replace("__", "_")
        # Lowercase
        clean_name = clean_name.lower().strip("_")
        
        # Avoid duplicate column names
        base_name = clean_name
        counter = 1
        while clean_name in new_cols.values():
            clean_name = f"{base_name}_{counter}"
            counter += 1
            
        new_cols[col] = clean_name
        
    df_cleaned.rename(columns=new_cols, inplace=True)
    return df_cleaned

def detect_outliers_iqr(df: pd.DataFrame, column: str, factor=1.5):
    """
    Detect outliers in a numeric column using the Interquartile Range (IQR) method.
    Returns a boolean series where True indicates an outlier.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return pd.Series(False, index=df.index)
    
    q25 = df[column].quantile(0.25)
    q75 = df[column].quantile(0.75)
    iqr = q75 - q25
    
    lower_bound = q25 - (factor * iqr)
    upper_bound = q75 + (factor * iqr)
    
    return (df[column] < lower_bound) | (df[column] > upper_bound)

def clean_dataset(df: pd.DataFrame, fill_missing=True, remove_duplicates=True, handle_outliers=False) -> tuple[pd.DataFrame, dict]:
    """
    Perform a complete cleaning pass over the dataset.
    Returns the cleaned DataFrame and a dictionary of operations performed.
    """
    df_cleaned = df.copy()
    report = {
        "rows_before": len(df),
        "duplicates_removed": 0,
        "missing_filled": {},
        "outliers_handled": {}
    }
    
    # 1. Remove duplicate records
    if remove_duplicates:
        initial_rows = len(df_cleaned)
        df_cleaned.drop_duplicates(inplace=True)
        report["duplicates_removed"] = initial_rows - len(df_cleaned)
        
    # 2. Handle missing values
    if fill_missing:
        for col in df_cleaned.columns:
            null_count = df_cleaned[col].isnull().sum()
            if null_count > 0:
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    median_val = df_cleaned[col].median()
                    df_cleaned[col] = df_cleaned[col].fillna(median_val)
                    report["missing_filled"][col] = f"Filled {null_count} missing values with median ({median_val})"
                else:
                    mode_series = df_cleaned[col].mode()
                    mode_val = mode_series.iloc[0] if not mode_series.empty else "Unknown"
                    df_cleaned[col] = df_cleaned[col].fillna(mode_val)
                    report["missing_filled"][col] = f"Filled {null_count} missing values with mode ('{mode_val}')"
                    
    # 3. Handle outliers (by clipping them to the IQR bounds)
    if handle_outliers:
        for col in df_cleaned.columns:
            if pd.api.types.is_numeric_dtype(df_cleaned[col]) and not col.lower().endswith("id"):
                outliers = detect_outliers_iqr(df_cleaned, col)
                outlier_count = outliers.sum()
                if outlier_count > 0:
                    q25 = df_cleaned[col].quantile(0.25)
                    q75 = df_cleaned[col].quantile(0.75)
                    iqr = q75 - q25
                    lower_bound = q25 - (1.5 * iqr)
                    upper_bound = q75 + (1.5 * iqr)
                    
                    # Clip values
                    df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
                    report["outliers_handled"][col] = f"Clipped {outlier_count} outliers to IQR bounds [{lower_bound:.2f}, {upper_bound:.2f}]"
                    
    return df_cleaned, report
