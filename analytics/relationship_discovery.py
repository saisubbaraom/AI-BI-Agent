import pandas as pd
import os

def clean_table_name(filename: str) -> str:
    """Removes extension and standard suffixes to get a clean entity name."""
    name, _ = os.path.splitext(filename)
    name = name.lower()
    # Remove plurals like customers -> customer
    if name.endswith("s") and len(name) > 3:
        name = name[:-1]
    return name

def discover_relationships(dfs: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Scans a dictionary of dataframes to find potential join relationships.
    Returns:
        List of dicts representing suggested joins:
        {
            "source_file": "customers.csv",
            "target_file": "orders.csv",
            "source_col": "customer_id",
            "target_col": "customer_id",
            "matching_pct": 95.2,
            "duplicate_warning": False,
            "join_type": "left"
        }
    """
    suggestions = []
    filenames = list(dfs.keys())
    
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            file_a = filenames[i]
            file_b = filenames[j]
            df_a = dfs[file_a]
            df_b = dfs[file_b]
            
            clean_a = clean_table_name(file_a)
            clean_b = clean_table_name(file_b)
            
            # Find candidate column pairs
            candidates = []
            for col_a in df_a.columns:
                col_a_clean = col_a.lower().replace("_", "").replace(" ", "")
                for col_b in df_b.columns:
                    col_b_clean = col_b.lower().replace("_", "").replace(" ", "")
                    
                    is_match = False
                    # Rule 1: Exact matching names (e.g. customer_id and customer_id)
                    if col_a_clean == col_b_clean and any(x in col_a_clean for x in ["id", "key", "code"]):
                        is_match = True
                    # Rule 2: ID column matching TableName_ID (e.g. customers.id and orders.customer_id)
                    elif col_a_clean == "id" and col_b_clean == f"{clean_a}id":
                        is_match = True
                    elif col_b_clean == "id" and col_a_clean == f"{clean_b}id":
                        is_match = True
                        
                    if is_match:
                        candidates.append((col_a, col_b))
                        
            # For each candidate column pair, calculate statistics
            for col_a, col_b in candidates:
                try:
                    set_a = set(df_a[col_a].dropna().astype(str).str.strip().str.lower())
                    set_b = set(df_b[col_b].dropna().astype(str).str.strip().str.lower())
                    
                    if not set_a or not set_b:
                        continue
                        
                    intersection = set_a.intersection(set_b)
                    
                    # Calculate coverage
                    coverage_a = (len(intersection) / len(set_a)) * 100 if len(set_a) > 0 else 0
                    coverage_b = (len(intersection) / len(set_b)) * 100 if len(set_b) > 0 else 0
                    
                    # Deduplicate warning: if key column has duplicates in both, it's many-to-many (high risk)
                    dup_a = df_a[col_a].duplicated().any()
                    dup_b = df_b[col_b].duplicated().any()
                    
                    # Recommend source vs target and join type
                    # Prefer mapping from a primary table (no duplicates) to a foreign table (has duplicates)
                    if dup_a and not dup_b:
                        # B is primary, A is foreign
                        src_file, tgt_file = file_b, file_a
                        src_col, tgt_col = col_b, col_a
                        matching_pct = coverage_a # % of foreign rows that match primary
                        join_type = "left" # Left join from primary (B) to foreign (A), or left join foreign (A) with primary (B)
                    else:
                        # A is primary or both have duplicates
                        src_file, tgt_file = file_a, file_b
                        src_col, tgt_col = col_a, col_b
                        matching_pct = coverage_b
                        join_type = "left"
                        
                    # Calculate overall match count
                    suggestions.append({
                        "source_file": src_file,
                        "target_file": tgt_file,
                        "source_col": src_col,
                        "target_col": tgt_col,
                        "matching_pct": round(matching_pct, 1),
                        "duplicate_warning": dup_a and dup_b,
                        "join_type": join_type
                    })
                except Exception as e:
                    print(f"Error evaluating candidate key {col_a} <-> {col_b}: {e}")
                    
    # Sort suggestions by matching percentage descending
    suggestions.sort(key=lambda x: x["matching_pct"], reverse=True)
    return suggestions

def calculate_join_stats(df_a: pd.DataFrame, df_b: pd.DataFrame, key_a: str, key_b: str, join_type: str) -> dict:
    """
    Calculates detailed preview statistics before merging df_a and df_b.
    """
    try:
        set_a = set(df_a[key_a].dropna().astype(str).str.strip().str.lower())
        set_b = set(df_b[key_b].dropna().astype(str).str.strip().str.lower())
        
        intersection = set_a.intersection(set_b)
        
        coverage_a = (len(intersection) / len(set_a)) * 100 if len(set_a) > 0 else 0
        coverage_b = (len(intersection) / len(set_b)) * 100 if len(set_b) > 0 else 0
        
        dup_a = df_a[key_a].duplicated().sum()
        dup_b = df_b[key_b].duplicated().sum()
        
        # Test merge row count
        temp_a = df_a[[key_a]].copy()
        temp_b = df_b[[key_b]].copy()
        temp_a["_src"] = 1
        temp_b["_tgt"] = 1
        
        test_merged = pd.merge(temp_a, temp_b, left_on=key_a, right_on=key_b, how=join_type)
        merged_row_count = len(test_merged)
        
        warning = ""
        if dup_a > 0 and dup_b > 0:
            warning = "⚠️ Many-to-many relationship detected. Merging will create duplicate/multiplied rows!"
        elif dup_a > 0 and join_type in ["right", "outer"]:
            warning = "⚠️ Joining on non-unique key might create unexpected row duplications."
        elif dup_b > 0 and join_type in ["left", "outer"]:
            warning = "⚠️ Joining on non-unique key might create unexpected row duplications."
            
        return {
            "row_count_a": len(df_a),
            "row_count_b": len(df_b),
            "coverage_a": round(coverage_a, 1),
            "coverage_b": round(coverage_b, 1),
            "duplicates_a": int(dup_a),
            "duplicates_b": int(dup_b),
            "merged_row_count": merged_row_count,
            "warning": warning
        }
    except Exception as e:
        return {
            "row_count_a": len(df_a),
            "row_count_b": len(df_b),
            "coverage_a": 0.0,
            "coverage_b": 0.0,
            "duplicates_a": 0,
            "duplicates_b": 0,
            "merged_row_count": 0,
            "warning": f"Error calculating preview: {str(e)}"
        }

def merge_datasets(df_a: pd.DataFrame, df_b: pd.DataFrame, key_a: str, key_b: str, join_type: str) -> pd.DataFrame:
    """
    Merges two dataframes on the specified keys.
    Handles duplicate columns by appending suffixes.
    """
    # Perform merge
    merged = pd.merge(df_a, df_b, left_on=key_a, right_on=key_b, how=join_type, suffixes=("", "_df_b"))
    
    # If the join key column in df_b was duplicated (e.g. key_b matches key_a name),
    # drop the duplicate column to keep the dataframe clean
    if key_a == key_b and f"{key_b}_df_b" in merged.columns:
        merged.drop(columns=[f"{key_b}_df_b"], inplace=True, errors="ignore")
        
    return merged
