import pandas as pd
import numpy as np
import json
import ast
from rag.vector_store import SimpleDocumentStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser

def parse_json_safely(text: str) -> dict:
    """Extracts and parses JSON safely from LLM output, with robust AST fallback."""
    text = text.strip()
    
    # Find the outer JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    
    if start == -1 or end == -1 or end < start:
        return {}
        
    json_str = text[start:end+1]
    
    try:
        return json.loads(json_str)
    except Exception:
        # Fallback to ast.literal_eval for single quotes/trailing commas
        try:
            parsed = ast.literal_eval(json_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
            
    return {}

def index_analysis_outputs(overview: dict, kpis: dict, insights: dict, recommendations: list[dict]) -> SimpleDocumentStore:
    """
    Compiles all analytical outputs into granular, structured documents
    and indexes them in the SimpleDocumentStore to enable token-efficient RAG.
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
    
    # 2. Index calculated KPIs individually
    for key, info in kpis.items():
        kpi_text = f"Calculated Business KPI - {info.get('label', key)}: {info.get('value')} (Format: {info.get('format')})"
        store.add_text(kpi_text, {"type": "kpi", "kpi_key": key})
    
    # 3. Index Narrative Insights individually
    exec_sum = insights.get("executive_summary", "")
    if exec_sum:
        store.add_text(f"Executive Summary:\n{exec_sum}", {"type": "executive_summary"})
        
    for finding in insights.get("key_findings", []):
        if finding:
            store.add_text(f"Key Finding: {finding}", {"type": "finding"})
            
    for trend in insights.get("trends_anomalies", []):
        if trend:
            store.add_text(f"Trend/Anomaly: {trend}", {"type": "trend"})
            
    for opp in insights.get("opportunities", []):
        if opp:
            store.add_text(f"Growth Opportunity: {opp}", {"type": "opportunity"})
            
    for risk in insights.get("risks", []):
        if risk:
            store.add_text(f"Risk: {risk}", {"type": "risk"})
    
    # 4. Index Recommendations individually
    for r in recommendations:
        rec_text = (
            f"Recommendation: {r.get('title')}\n"
            f"- Priority: {r.get('priority')}\n"
            f"- Impact: {r.get('impact')}\n"
            f"- Confidence: {r.get('confidence')}%\n"
            f"- Description: {r.get('description')}"
        )
        store.add_text(rec_text, {"type": "recommendation"})
    
    # Build index
    store.build_index()
    return store

class StructuredPandasQueryEngine:
    """
    Safely translates natural language queries into structured JSON database operations,
    runs them using Pandas, and formats results.
    """
    def __init__(self, df: pd.DataFrame, llm):
        self.df = df.copy()
        self.llm = llm
        
        # Auto-compute profit column if revenue/cost or revenue_generated/spend are present
        if "revenue" in self.df.columns and "cost" in self.df.columns:
            if "profit" not in self.df.columns:
                self.df["profit"] = self.df["revenue"] - self.df["cost"]
        elif "revenue_generated" in self.df.columns and "spend" in self.df.columns:
            if "profit" not in self.df.columns:
                self.df["profit"] = self.df["revenue_generated"] - self.df["spend"]

    def generate_query_plan(self, question: str, chat_history: list = None) -> dict:
        """Prompts Grok LLM to generate a structured JSON query plan, using recent chat context if provided."""
        if not self.llm:
            return {}
            
        columns = list(self.df.columns)
        dtypes = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        sample_data = self.df.head(3).to_dict(orient="records")

        # Format recent chat history context for the query planner
        recent_chat_context = ""
        if chat_history:
            recent_turns = []
            for msg in chat_history[-3:]:
                role = msg.get("role", "user").upper()
                content = msg.get("content", "")
                recent_turns.append(f"{role}: {content}")
            recent_chat_context = "Recent Conversation History:\n" + "\n".join(recent_turns) + "\n\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a translation layer between a user question and a structured data query.\n"
                "Review the dataset columns, data types, and sample data. "
                "Determine if the user's question requires a calculations search or a data aggregation "
                "on the Pandas DataFrame.\n\n"
                "CRITICAL RULES FOR QUERY PLANNING:\n"
                "1. CHOOSE THE RIGHT query_type:\n"
                "   - Use \"aggregation\" if the question is comparative, asking for rankings, categories, or groups (e.g., \"which brand has highest sales\", \"top products\", \"breakdown by location\"). Always set \"groupby_column\" to the categorical dimension (e.g. \"product_category\") and \"agg_column\" to the metric column (e.g. \"revenue\", \"profit\") and \"agg_func\" to \"sum\" or \"mean\".\n"
                "   - Use \"general_stats\" if the question asks for a single aggregate metric across the whole dataset (or a filtered subset) without grouping (e.g., \"what is the total revenue\", \"average order cost in New York\"). Leave \"groupby_column\" as null.\n"
                "   - Use \"filter_only\" if the user wants to list/see individual records/rows (e.g., \"show me the first 5 records\", \"list orders from Texas\"). Do not set \"groupby_column\".\n"
                "2. NO ARBITRARY FILTERS:\n"
                "   - ONLY apply filters (\"filter_column\", \"filter_op\", \"filter_value\") if the user explicitly specifies a filtering condition in the question (e.g. \"in Washington\", \"age > 50\").\n"
                "   - NEVER invent filters or pick values from the sample data (like filtering by \"Laptops\" or \"Washington\") unless the user specifically asked for them.\n"
                "3. TERM MAPPING & AGGREGATION FUNCTIONS:\n"
                "   - If the user asks for 'brand', 'product', or 'item', map it to the categorical column representing items (e.g. 'product_category', 'campaign_name', etc.).\n"
                "   - If the user asks for 'profit', 'earnings', or 'margin', map it to 'profit'.\n"
                "   - If the user asks for 'sales', map it to 'revenue' (preferable) or 'quantity'.\n"
                "   - Default to 'sum' as the aggregation function for volume/total metrics (revenue, quantity, profit) unless average/mean/median is explicitly requested. Do NOT confuse conversational words like 'means', 'meaning', or 'I mean' with the mathematical 'mean' (average) function.\n\n"
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
                "{recent_chat}"
                "Current Question: {question}"
            ))
        ])
        
        chain = prompt | self.llm
        try:
            res = chain.invoke({
                "columns": columns,
                "dtypes": dtypes,
                "sample": json.dumps(sample_data),
                "recent_chat": recent_chat_context,
                "question": question
            })
            content = res.content if hasattr(res, "content") else str(res)
            plan = parse_json_safely(content)
            
            # Post-processing safeguard for conversational "mean" (e.g. "means which made profits high")
            if plan and plan.get("agg_func") == "mean":
                q_lower = question.lower()
                # Indicators of actual math averages
                avg_indicators = ["average", "avg", "mean of", "mean profit", "mean revenue", "mean cost", "mean spend", "mean value", "mathematical mean"]
                has_avg_indicator = any(ind in q_lower for ind in avg_indicators)
                
                # Conversational "means" or "meaning" indicators
                conversational_indicators = ["means", "meaning", "i mean", "which means"]
                has_conv_indicator = any(ind in q_lower for ind in conversational_indicators)
                
                if has_conv_indicator and not has_avg_indicator:
                    print(f"DEBUG: Overriding mathematical 'mean' to 'sum' due to conversational keywords in question: '{question}'")
                    plan["agg_func"] = "sum"
                    
            return plan
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
            
            sort_asc = plan.get("sort_ascending")
            if sort_asc is None:
                sort_asc = False
                
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
            elif query_type == "general_stats":
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
