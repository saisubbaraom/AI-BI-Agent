import pandas as pd
import numpy as np
import json
from rag.vector_store import SimpleDocumentStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser

def index_analysis_outputs(overview: dict, kpis: dict, insights: dict, recommendations: list[dict]) -> SimpleDocumentStore:
    """
    Compiles all analytical outputs into structured documents and indexes them in the SimpleDocumentStore.
    """
    store = SimpleDocumentStore()
    
    # 1. Index Dataset Overview & Schema
    schema_lines = [f"- {col}: {dtype}" for col, dtype in overview.get("dtypes", {}).items()]
    overview_text = (
        "Dataset Structural Overview:\n"
        f"- Total Columns: {overview.get('num_columns')}\n"
        f"- Total Rows: {overview.get('num_rows')}\n"
        f"- Duplicate Rows: {overview.get('duplicate_rows')}\n"
        "Columns and Data Types:\n" + "\n".join(schema_lines)
    )
    store.add_text(overview_text, {"type": "metadata"})
    
    # 2. Index calculated KPIs
    kpi_lines = []
    for key, info in kpis.items():
        kpi_lines.append(f"- {info['label']}: {info['value']} (Format: {info['format']})")
    kpis_text = "Calculated Business KPIs:\n" + "\n".join(kpi_lines)
    store.add_text(kpis_text, {"type": "kpis"})
    
    # 3. Index Narrative Insights
    exec_sum = insights.get("executive_summary", "")
    findings = "\n".join([f"- {f}" for f in insights.get("key_findings", [])])
    trends = "\n".join([f"- {t}" for t in insights.get("trends_anomalies", [])])
    opps = "\n".join([f"- {o}" for o in insights.get("opportunities", [])])
    risks = "\n".join([f"- {r}" for r in insights.get("risks", [])])
    
    insights_text = (
        f"Executive Summary:\n{exec_sum}\n\n"
        f"Key Findings:\n{findings}\n\n"
        f"Trends & Anomalies:\n{trends}\n\n"
        f"Growth Opportunities:\n{opps}\n\n"
        f"Risks:\n{risks}"
    )
    store.add_text(insights_text, {"type": "insights"})
    
    # 4. Index Recommendations
    rec_lines = []
    for r in recommendations:
        rec_lines.append(
            f"Recommendation: {r.get('title')}\n"
            f"- Priority: {r.get('priority')}\n"
            f"- Impact: {r.get('impact')}\n"
            f"- Confidence: {r.get('confidence')}%\n"
            f"- Description: {r.get('description')}\n"
        )
    recs_text = "Strategic Recommendations:\n\n" + "\n\n".join(rec_lines)
    store.add_text(recs_text, {"type": "recommendations"})
    
    # Build index
    store.build_index()
    return store

class StructuredPandasQueryEngine:
    """
    Safely translates natural language queries into structured JSON database operations,
    runs them using Pandas, and formats results.
    """
    def __init__(self, df: pd.DataFrame, llm):
        self.df = df
        self.llm = llm

    def generate_query_plan(self, question: str) -> dict:
        """Prompts Grok LLM to generate a structured JSON query plan."""
        if not self.llm:
            return {}
            
        columns = list(self.df.columns)
        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        sample_data = self.df.head(3).to_dict(orient="records")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a translation layer between a user question and a structured data query.\n"
                "Review the dataset columns, data types, and sample data. "
                "Determine if the user's question requires a calculations search or a data aggregation "
                "on the Pandas DataFrame.\n\n"
                "If it does, respond with a valid JSON query plan. "
                "If the question is general or does not require a data query, set query_needed to false.\n\n"
                "Return ONLY the raw JSON object. Do not wrap in markdown tags:\n"
                "{{\n"
                '  "query_needed": true,\n'
                '  "query_type": "aggregation" | "filter_only" | "general_stats",\n'
                '  "groupby_column": "column_name_or_null",\n'
                '  "agg_column": "column_name_or_null",\n'
                '  "agg_func": "sum" | "mean" | "count" | "min" | "max" | "median",\n'
                '  "filter_column": "column_name_or_null",\n'
                '  "filter_op": "eq" | "gt" | "lt" | "contains" | "between",\n'
                '  "filter_value": "value_or_list_of_values",\n'
                '  "sort_column": "column_name_or_null",\n'
                '  "sort_ascending": true | false,\n'
                '  "limit": 5\n'
                "}}\n"
                "Ensure that column names match the dataset columns exactly."
            )),
            ("user", (
                "Columns: {columns}\n"
                "Data Types: {dtypes}\n"
                "Sample Rows: {sample}\n\n"
                "Question: {question}"
            ))
        ])
        
        chain = prompt | self.llm | SimpleJsonOutputParser()
        try:
            return chain.invoke({
                "columns": columns,
                "dtypes": dtypes,
                "sample": json.dumps(sample_data),
                "question": question
            })
        except Exception as e:
            print(f"Error generating query plan: {e}")
            return {}

    def execute_query_plan(self, plan: dict) -> str:
        """Safely executes the JSON plan on the Pandas DataFrame and formats the output."""
        if not plan or not plan.get("query_needed", False):
            return "No query plan generated."
            
        try:
            df_temp = self.df.copy()
            filter_col = plan.get("filter_column")
            filter_op = plan.get("filter_op")
            filter_val = plan.get("filter_value")
            
            # 1. Apply Filters
            if filter_col and filter_col in df_temp.columns and filter_op:
                if filter_op == "eq":
                    df_temp = df_temp[df_temp[filter_col].astype(str).str.lower() == str(filter_val).lower()]
                elif filter_op == "gt":
                    df_temp = df_temp[df_temp[filter_col] > float(filter_val)]
                elif filter_op == "lt":
                    df_temp = df_temp[df_temp[filter_col] < float(filter_val)]
                elif filter_op == "contains":
                    df_temp = df_temp[df_temp[filter_col].astype(str).str.contains(str(filter_val), case=False, na=False)]
                elif filter_op == "between" and isinstance(filter_val, list) and len(filter_val) == 2:
                    # Handle dates or numeric
                    if pd.api.types.is_numeric_dtype(df_temp[filter_col]):
                        df_temp = df_temp[df_temp[filter_col].between(float(filter_val[0]), float(filter_val[1]))]
                    else:
                        df_temp = df_temp[df_temp[filter_col].between(str(filter_val[0]), str(filter_val[1]))]
            
            if df_temp.empty:
                return "The query returned no results matching the filter criteria."
                
            query_type = plan.get("query_type")
            groupby_col = plan.get("groupby_column")
            agg_col = plan.get("agg_column")
            agg_func = plan.get("agg_func", "sum")
            sort_col = plan.get("sort_column")
            sort_asc = plan.get("sort_ascending", False)
            limit = plan.get("limit", 5)
            if limit is None or not isinstance(limit, int):
                limit = 10
            else:
                limit = min(limit, 10)
            
            # 2. Perform Calculations / Aggregations
            result_df = pd.DataFrame()
            
            if query_type == "aggregation" and groupby_col and groupby_col in df_temp.columns:
                if agg_col and agg_col in df_temp.columns:
                    grouped = df_temp.groupby(groupby_col)[agg_col].agg(agg_func).reset_index()
                    result_df = grouped
                else:
                    # Group by and count rows
                    grouped = df_temp.groupby(groupby_col).size().reset_index(name="count")
                    result_df = grouped
                    agg_col = "count"
            elif query_type == "general_stats" or not groupby_col:
                if agg_col and agg_col in df_temp.columns:
                    val = df_temp[agg_col].agg(agg_func)
                    return f"Calculation Result: {agg_func.upper()} of '{agg_col}' is {val:.2f} (based on filtered criteria)."
                else:
                    return f"Overview Stats: The filtered dataset contains {len(df_temp)} matching records."
            else:
                result_df = df_temp
                
            # 3. Apply Sorting
            if sort_col and sort_col in result_df.columns:
                result_df = result_df.sort_values(by=sort_col, ascending=sort_asc)
            elif agg_col in result_df.columns:
                # Default sorting by aggregated metric
                result_df = result_df.sort_values(by=agg_col, ascending=sort_asc)
                
            # 4. Apply Limit
            if limit:
                result_df = result_df.head(limit)
                
            # Convert result to a markdown table or text summary
            try:
                md_table = result_df.to_markdown(index=False)
            except Exception:
                md_table = result_df.to_string(index=False)
            return f"Queried Data Results (Top {limit}):\n\n{md_table}"
            
        except Exception as e:
            return f"Error executing data query: {str(e)}"
