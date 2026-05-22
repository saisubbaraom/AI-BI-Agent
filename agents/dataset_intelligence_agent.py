import pandas as pd
import numpy as np
import json
import re
from agents.base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from rag.retrieval import parse_json_safely

class DatasetIntelligenceAgent(BaseAgent):
    """
    Dataset Intelligence Agent: Automatically analyzes dataset schema, profiling data,
    and statistical distributions to infer dataset theme, detect PII/IDs, and dynamically
    propose KPIs and visualizations.
    """
    def __init__(self):
        super().__init__(
            name="Dataset Intelligence Agent",
            role="Specialist in profiling datasets, identifying data domains, detecting PII, and planning dynamic KPIs & visualizations."
        )

    def detect_pii_and_ids(self, df: pd.DataFrame) -> dict:
        """
        Heuristically scan column names and sample values to detect PII and ID fields.
        """
        pii_cols = []
        id_cols = []
        
        pii_keywords = ["name", "phone", "email", "address", "ssn", "social_security", "dob", "birth", "mobile", "zip", "password", "passport"]
        id_keywords = ["id", "key", "code", "number", "no"]
        
        for col in df.columns:
            col_lower = col.lower()
            col_clean = col_lower.replace("_", "").replace(" ", "")
            
            # Check PII
            is_pii = False
            for kw in pii_keywords:
                if kw in col_clean:
                    pii_cols.append(col)
                    is_pii = True
                    break
                    
            if not is_pii:
                # Check ID
                for kw in id_keywords:
                    # Match ends with, starts with, or exact matches
                    if col_clean.endswith(kw) or col_clean.startswith(kw) or col_clean == kw:
                        id_cols.append(col)
                        break
                        
        return {
            "pii": pii_cols,
            "ids": id_cols
        }

    def evaluate_kpi_expression(self, df: pd.DataFrame, kpi_def: dict) -> float:
        """
        Safely evaluates a KPI expression on the DataFrame, with fallback parameters.
        """
        expr = kpi_def.get("expression", "")
        label = kpi_def.get("label", "Metric")
        
        # Safe environment
        safe_env = {
            "df": df,
            "pd": pd,
            "np": np,
            "__builtins__": {}
        }
        
        try:
            # Basic validation: ensure expression is string and references 'df'
            if not isinstance(expr, str) or "df" not in expr:
                raise ValueError("Invalid expression format")
                
            # Safely evaluate
            val = eval(expr, safe_env)
            
            # Unwrap Series, Index, or numpy types
            if isinstance(val, (pd.Series, pd.Index)):
                val = val.iloc[0] if not val.empty else 0
            elif isinstance(val, np.generic):
                val = val.item()
                
            if pd.isna(val) or (isinstance(val, float) and np.isnan(val)):
                return 0.0
                
            return val
            
        except Exception as e:
            # Fallback based on metadata
            column = kpi_def.get("column")
            func = kpi_def.get("func", "count")
            
            print(f"Failed to evaluate expression '{expr}' for KPI '{label}'. Error: {e}. Using fallback: {func}({column})")
            
            if column and column in df.columns:
                try:
                    if func == "sum":
                        return float(df[column].sum())
                    elif func == "mean":
                        return float(df[column].mean())
                    elif func == "nunique":
                        return int(df[column].nunique())
                    elif func == "max":
                        return float(df[column].max())
                    elif func == "min":
                        return float(df[column].min())
                    elif func == "count":
                        return int(df[column].count())
                except Exception as ex:
                    print(f"Fallback calculation failed for '{label}': {ex}")
            
            # Final fallback
            if func == "count" or not column:
                return len(df)
            return 0.0

    def run(self, df: pd.DataFrame) -> dict:
        """
        Profiles the dataset and uses LLM to generate dynamic KPI and visualization definitions.
        Calculates the KPIs on the dataset.
        """
        # 1. Run local heuristics
        heuristic_res = self.detect_pii_and_ids(df)
        
        # 2. Gather profiling information to send to LLM
        columns = list(df.columns)
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        num_rows = len(df)
        num_cols = len(df.columns)
        
        # Sample values (up to 3 rows)
        sample_rows = df.head(3).to_dict(orient="records")
        
        # Column statistics (min, max, null percent, unique count)
        stats = {}
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = round((null_count / num_rows) * 100, 2) if num_rows > 0 else 0
            unique_cnt = df[col].nunique()
            
            col_stats = {
                "null_percent": null_pct,
                "unique_count": unique_cnt
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats["min"] = float(df[col].min()) if not pd.isna(df[col].min()) else "N/A"
                col_stats["max"] = float(df[col].max()) if not pd.isna(df[col].max()) else "N/A"
                col_stats["mean"] = float(df[col].mean()) if not pd.isna(df[col].mean()) else "N/A"
                
            stats[col] = col_stats

        # 3. Request LLM Analysis
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert AI Data Scientist and Dataset Analyst.\n"
                "Your role is to analyze a dataset's metadata (columns, data types, statistics, and sample rows) "
                "and infer its core business or domain theme (e.g. 'EV Vehicles Dataset', 'Retail Product Sales', 'HR Workforce').\n\n"
                "You must suggest:\n"
                "1. A clean domain theme and description.\n"
                "2. 4 to 6 relevant business KPIs. For each KPI, provide a valid, safe, single-line Pandas expression "
                "that calculates the KPI from a DataFrame variable named 'df'. This expression will be evaluated directly "
                "using python's eval(). Ensure expressions are safe and syntactically correct.\n"
                "3. 3 to 5 logical visualizations tailored to the dataset. Suggest the chart type, x_col, and optional y_col / color_col.\n\n"
                "CRITICAL RULES FOR PANDAS EXPRESSIONS:\n"
                "- Write simple, safe expressions, like: `df.shape[0]`, `df['electric_range'].mean()`, `df[df['ev_type'] == 'BEV'].shape[0]`, `df['cogs'].sum()`\n"
                "- Handle missing or division by zero safely using standard pandas methods if necessary.\n"
                "- Refer to columns exactly as defined in the schema. Do not invent columns.\n\n"
                "Return ONLY a valid JSON object matching the following schema. "
                "Do not wrap in markdown ```json blocks:\n"
                "{{\n"
                '  "theme": "EV Vehicles Dataset",\n'
                '  "domain": "Electric Vehicles",\n'
                '  "description": "Analysis of electric vehicles registrations, ranges, and adoption.",\n'
                '  "suggested_kpis": [\n'
                '    {{\n'
                '      "id": "total_vehicles",\n'
                '      "label": "Total EV Vehicles",\n'
                '      "expression": "df.shape[0]",\n'
                '      "func": "count",\n'
                '      "column": null,\n'
                '      "format": "integer",\n'
                '      "description": "Total number of electric vehicles registered."\n'
                '    }},\n'
                '    {{\n'
                '      "id": "avg_range",\n'
                '      "label": "Average Range",\n'
                '      "expression": "df[\'electric_range\'].mean()",\n'
                '      "func": "mean",\n'
                '      "column": "electric_range",\n'
                '      "format": "number",\n'
                '      "description": "Average range of the vehicles in miles."\n'
                '    }}\n'
                '  ],\n'
                '  "suggested_visualizations": [\n'
                '    {{\n'
                '      "id": "mfr_dist",\n'
                '      "title": "Manufacturer Distribution",\n'
                '      "type": "bar",\n'
                '      "x_col": "manufacturer",\n'
                '      "y_col": null,\n'
                '      "color_col": null,\n'
                '      "description": "Comparison of vehicle counts by manufacturer."\n'
                '    }}\n'
                '  ]\n'
                "}}\n"
            )),
            ("user", (
                "Dataset Overview:\n"
                "- Number of Rows: {num_rows}\n"
                "- Number of Columns: {num_cols}\n"
                "Data Types:\n{dtypes}\n"
                "Column Stats:\n{stats}\n"
                "Sample Rows:\n{sample_rows}\n"
            ))
        ])

        try:
            chain = prompt | self.llm
            res = chain.invoke({
                "num_rows": num_rows,
                "num_cols": num_cols,
                "dtypes": json.dumps(dtypes, indent=2),
                "stats": json.dumps(stats, indent=2),
                "sample_rows": json.dumps(sample_rows, indent=2)
            })
            
            content = res.content if hasattr(res, "content") else str(res)
            parsed_res = parse_json_safely(content)
            
            theme = parsed_res.get("theme", "Generic Dataset")
            domain = parsed_res.get("domain", "Generic")
            description = parsed_res.get("description", f"Automated analysis of {num_rows} records and {num_cols} features.")
            suggested_kpis = parsed_res.get("suggested_kpis", [])
            suggested_visualizations = parsed_res.get("suggested_visualizations", [])
            
        except Exception as e:
            print(f"Error calling LLM in DatasetIntelligenceAgent: {e}")
            # Fallback definitions
            theme = "Generic Dataset"
            domain = "Generic"
            description = f"Automated analysis of {num_rows} records."
            suggested_kpis = [
                {
                    "id": "row_count",
                    "label": "Total Records",
                    "expression": "df.shape[0]",
                    "func": "count",
                    "column": None,
                    "format": "integer",
                    "description": "Total number of records in the dataset."
                }
            ]
            
            # Simple numeric column KPIs
            num_cols_list = df.select_dtypes(include=[np.number]).columns
            num_cols_list = [c for c in num_cols_list if not c.lower().endswith("id")]
            if num_cols_list:
                for c in num_cols_list[:2]:
                    suggested_kpis.append({
                        "id": f"avg_{c}",
                        "label": f"Average {c.replace('_', ' ').title()}",
                        "expression": f"df['{c}'].mean()",
                        "func": "mean",
                        "column": c,
                        "format": "number",
                        "description": f"Average value of {c}."
                    })
            suggested_visualizations = []

        # 4. Calculate computed KPIs using the helper
        computed_kpis = {}
        for kpi in suggested_kpis:
            val = self.evaluate_kpi_expression(df, kpi)
            computed_kpis[kpi["id"]] = {
                "label": kpi["label"],
                "value": val,
                "format": kpi["format"],
                "description": kpi.get("description", "")
            }

        return {
            "theme": theme,
            "domain": domain,
            "description": description,
            "kpis": computed_kpis,
            "suggested_charts": suggested_visualizations,
            "detected_pii": heuristic_res["pii"],
            "detected_ids": heuristic_res["ids"],
            "raw_kpi_defs": suggested_kpis
        }
