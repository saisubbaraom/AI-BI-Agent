
import streamlit as st
import pandas as pd
import numpy as np
import os
import io



from utils.config import is_api_configured
from agents.chat_agent import ChatAgent
from utils.helpers import (
    inject_premium_css, render_kpi_card, render_recommendation_card,
    format_number, format_currency, format_percent, summarize_chat_history
)
from utils.data_cleaner import normalize_column_names
from agents.master_agent import MasterAgent
from reports.pdf_generator import generate_pdf_report
from reports.docx_generator import generate_docx_report

# 1. Page Configuration
st.set_page_config(
    page_title="Business Analyst AI Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
import time

# Always wipe chat_history on every page load — prevents stale API error messages from displaying
# Use process-based stamp so every server restart forces full session clear
_SERVER_STAMP = str(os.getpid())
if st.session_state.get("_server_pid") != _SERVER_STAMP:
    st.session_state.clear()
    st.session_state["_server_pid"] = _SERVER_STAMP

# Initialize Session States
if "analysis_bundle" not in st.session_state:
    st.session_state["analysis_bundle"] = None
if "df_raw" not in st.session_state:
    st.session_state["df_raw"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

# Inject Styles
inject_premium_css()

# 2. Sidebar Configuration
with st.sidebar:
    st.markdown('<div class="section-header">🤖 LLM Model Selection</div>', unsafe_allow_html=True)
    
    available_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "qwen/qwen3-32b",
        "meta-llama/llama-4-scout-17b-16e-instruct"
    ]
    
    if "selected_model" not in st.session_state:
        st.session_state["selected_model"] = "llama-3.3-70b-versatile"
        
    try:
        current_idx = available_models.index(st.session_state["selected_model"])
    except ValueError:
        current_idx = 0
        
    selected_model = st.selectbox(
        "Choose Model",
        options=available_models,
        index=current_idx,
        help="Select which LLM model to use. If you encounter Rate Limit (429) errors, switch to llama-3.1-8b-instant which has a higher rate limit quota."
    )
    
    if selected_model != st.session_state["selected_model"]:
        st.session_state["selected_model"] = selected_model
        st.toast(f"Switched model to {selected_model}")
        
    st.info("💡 **Auto-Fallback**: If the primary model hits a daily rate limit (429), it will automatically fall back to **llama-3.1-8b-instant** to complete your analysis.")

    if st.button("🔄 Clear App Cache / Reset", use_container_width=True, help="Clear session cache and reload the application."):
        st.session_state.clear()
        st.rerun()
        
    st.markdown("---")
    st.markdown('<div class="section-header">🧹 Data Processing Options</div>', unsafe_allow_html=True)
    
    clean_data = st.checkbox("Enable Automated Data Cleaning", value=True, help="Auto-imputes nulls, drops duplicates, and handles outliers.")
    
    if clean_data:
        fill_missing = st.checkbox("Impute Missing Values", value=True, help="Fill numerical columns with median, categorical with mode.")
        remove_duplicates = st.checkbox("Remove Duplicate Records", value=True, help="Drop exact duplicates.")
        handle_outliers = st.checkbox("Clip Numerical Outliers", value=False, help="Clip extreme values to 1.5x IQR boundaries.")
    else:
        fill_missing = False
        remove_duplicates = False
        handle_outliers = False
        
    st.markdown("---")
    st.markdown("### About Business Analyst AI")
    st.caption(
        "This platform acts as a senior strategy consultant. It automatically profiles data quality, "
        "calculates domain KPIs, builds visualizations, writes executive reports via Grok LLM, "
        "and answers analytics queries in the RAG Chat."
    )

# 3. Main Header
st.markdown('<div class="main-title">Business Analyst AI Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload your raw business data and receive executive-level summaries, KPIs, interactive charts, and actionable strategic recommendations instantly.</div>', unsafe_allow_html=True)

# 4. File Upload Section
uploaded_file = st.file_uploader(
    "Upload dataset (CSV, XLSX, XLS, or Parquet)",
    type=["csv", "xlsx", "xls", "parquet"]
)

if uploaded_file is not None:

    # Check if a new file is uploaded
    file_name = uploaded_file.name

    if st.session_state["last_uploaded_name"] != file_name:
        st.session_state["analysis_bundle"] = None
        st.session_state["df_raw"] = None
        st.session_state["chat_history"] = []
        st.session_state["last_uploaded_name"] = file_name

    # Load file only once
    if st.session_state["df_raw"] is None:

        with st.spinner("Reading dataset file..."):

            file_ext = os.path.splitext(file_name)[1].lower()

            try:

                if file_ext == ".csv":

                    try:
                        uploaded_file.seek(0)
                        df = pd.read_csv(uploaded_file)

                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file,
                            encoding="latin1"
                        )

                elif file_ext in [".xlsx", ".xls"]:

                    uploaded_file.seek(0)

                    df = pd.read_excel(
                        uploaded_file,
                        engine="openpyxl"
                    )

                elif file_ext == ".parquet":

                    uploaded_file.seek(0)

                    df = pd.read_parquet(uploaded_file)

                else:
                    st.error(
                        f"Unsupported file format: {file_ext}"
                    )
                    st.stop()

                if df.empty:
                    st.error("Uploaded dataset is empty.")
                    st.stop()

                st.session_state["df_raw"] = df
                st.toast("🎉 Dataset loaded successfully!", icon="✅")

            except Exception:
                import traceback

                st.error(
                    f"""
FULL TRACEBACK:

{traceback.format_exc()}
"""
                )

                st.stop()

    df_raw = st.session_state["df_raw"]
    
    # Persistent dataset info to keep page layout/DOM identical across reruns (prevents st.tabs resetting)
    st.success(f"📂 **Active Dataset:** `{file_name}` | `{df_raw.shape[0]:,}` rows × `{df_raw.shape[1]:,}` columns")
    
    # Run analysis pipeline if not already run
    if st.session_state["analysis_bundle"] is None:
        with st.spinner("Initializing Multi-Agent Pipeline... (Profiling, KPIs, Grok Insights & Action Plans)"):
            # Clean column names first for internal aggregates
            df_norm = normalize_column_names(df_raw)
            
            master = MasterAgent()
            try:
                bundle = master.run(
                    df_norm,
                    clean_data=clean_data,
                    fill_missing=fill_missing,
                    remove_duplicates=remove_duplicates,
                    handle_outliers=handle_outliers
                )
                st.session_state["analysis_bundle"] = bundle
                st.session_state["dataset_context"] = {
                    "shape": bundle["df"].shape,
                    "columns": bundle["df"].columns.tolist(),
                    "dtypes": bundle["df"].dtypes.astype(str).to_dict()
                }
            except Exception as e:
                st.error(f"Pipeline error: {str(e)}")
                st.stop()
                
    bundle = st.session_state["analysis_bundle"]
    if bundle is not None and "dataset_context" not in st.session_state:
        st.session_state["dataset_context"] = {
            "shape": bundle["df"].shape,
            "columns": bundle["df"].columns.tolist(),
            "dtypes": bundle["df"].dtypes.astype(str).to_dict()
        }
    
    # Sanitize insights dictionary to guarantee keys are iterable lists and never None/missing
    insights = bundle.get("insights") or {}
    if not isinstance(insights, dict):
        insights = {}
    for k in ["key_findings", "trends_anomalies", "opportunities", "risks"]:
        val = insights.get(k)
        if isinstance(val, list):
            insights[k] = [str(x) for x in val if x is not None]
        elif isinstance(val, str):
            insights[k] = [val]
        else:
            insights[k] = []
    if not insights.get("executive_summary"):
        insights["executive_summary"] = "No executive summary available."
    bundle["insights"] = insights
    
    # 5. Render Tabs
    tab_overview, tab_kpi, tab_viz, tab_insights, tab_recs, tab_chat, tab_export = st.tabs([
        "📤 Data Overview",
        "📊 KPI Dashboard",
        "📈 Visualizations",
        "💡 AI Insights",
        "🛠️ Action Recommendations",
        "💬 AI Consultant Chat",
        "📥 Report Export"
    ])
    
    # --- TAB 1: DATA OVERVIEW ---
    with tab_overview:
        st.markdown('<div class="section-header">Dataset Overview & Health Audit</div>', unsafe_allow_html=True)
        
        overview = bundle["overview"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{overview['num_rows']:,}")
        col2.metric("Total Columns", f"{overview['num_columns']}")
        col3.metric("Duplicate Rows", f"{overview['duplicate_rows']}")
        col4.metric("Memory Usage", f"{overview['memory_usage_mb']} MB")
        
        # Display Cleaning Logs if applicable
        if clean_data and bundle["cleaning_logs"]:
            logs = bundle["cleaning_logs"]
            with st.expander("📝 View Data Cleaning Operations Done", expanded=False):
                st.markdown(f"**Duplicate rows removed:** {logs['duplicates_removed']}")
                if logs["missing_filled"]:
                    st.markdown("**Imputed Missing Values:**")
                    for c, log in logs["missing_filled"].items():
                        st.markdown(f"- `{c}`: {log}")
                if logs["outliers_handled"]:
                    st.markdown("**Outliers Handled:**")
                    for c, log in logs["outliers_handled"].items():
                        st.markdown(f"- `{c}`: {log}")
                        
        st.markdown("### Data Quality Audit")
        quality = bundle["quality_report"]
        q_data = []
        for col, q in quality.items():
            q_data.append({
                "Column Name": col,
                "Data Type": q["dtype"],
                "Missing Values": q["null_count"],
                "Missing %": f"{q['null_percent']}%",
                "Outliers (IQR)": q["outliers_count"] if q["is_numeric"] else "N/A"
            })
        st.dataframe(pd.DataFrame(q_data), use_container_width=True)
        
        st.markdown("### Raw Data Preview (First 15 rows)")
        st.dataframe(bundle["df"].head(15), use_container_width=True)
        
        # Show demographics sub-panel if detected
        demographics = overview.get("demographics", {})
        if demographics:
            st.markdown("### Detected Customer Demographics")
            demo_cols = st.columns(len(demographics))
            for idx, (demo_name, details) in enumerate(demographics.items()):
                with demo_cols[idx]:
                    st.markdown(f"**{demo_name.title()}** (Column: `{details['column']}`)")
                    if demo_name in ["age", "income"] and "mean" in details:
                        st.markdown(f"- Average: **{details['mean']:,}**")
                        st.markdown(f"- Median: **{details['median']:,}**")
                    
                    # Renders a tiny summary table
                    dist_df = pd.DataFrame(list(details["distribution"].items()), columns=["Segment", "Percentage"])
                    st.dataframe(dist_df, hide_index=True, use_container_width=True)
                    
    # --- TAB 2: KPI DASHBOARD ---
    with tab_kpi:
        st.markdown(f'<div class="section-header">Executive Dashboard: {bundle["domain"]} KPIs</div>', unsafe_allow_html=True)
        
        kpis = bundle["kpis"]
        
        # Render KPI Cards in columns
        kpi_keys = list(kpis.keys())
        kpi_cols = st.columns(min(len(kpi_keys), 4))
        
        domain_class = bundle["domain"].lower()
        
        for idx, key in enumerate(kpi_keys):
            col_idx = idx % 4
            info = kpis[key]
            val = info["value"]
            fmt = info["format"]
            
          # Format value safely
            try:

                if pd.isna(val):
                    val_str = "N/A"

                elif fmt == "currency":
                    val_str = format_currency(float(val))

                elif fmt == "percent":
                    val_str = format_percent(float(val))

                elif fmt == "integer":
                    val_str = f"{int(float(val)):,}"

                elif fmt == "number":
                    val_str = format_number(float(val))

                else:
                    val_str = str(val)

            except Exception:
                val_str = str(val)

            with kpi_cols[col_idx]:
                render_kpi_card(
                    title=info["label"],
                    value=val_str,
                    domain_class=domain_class
                )
                
        # Display KPI Table data below
        st.markdown("### KPI Summary Table")
        kpi_table_data = []
        for key, info in kpis.items():
            val = info["value"]
            fmt = info["format"]
            if fmt == "currency":
                val_str = format_currency(val)
            elif fmt == "percent":
                val_str = format_percent(val)
            elif fmt == "integer":
                try:
                    val_str = f"{int(float(val)):,}"
                except:
                    val_str = str(val)
            else:
                val_str = format_number(val)
            kpi_table_data.append({
                "Metric Name": info["label"],
                "Calculated Value": val_str,
                "Raw Decimal": val
            })
        st.dataframe(pd.DataFrame(kpi_table_data), use_container_width=True, hide_index=True)
        
    # --- TAB 3: AUTO VISUALIZATIONS & CUSTOM BUILDER ---
    with tab_viz:
        st.markdown('<div class="section-header">Automated Insights Charts</div>', unsafe_allow_html=True)
        
        charts = bundle["charts"]
        if charts:
            for c_key, c_info in charts.items():
                with st.expander(f"📊 {c_info['title']}", expanded=True):
                    st.plotly_chart(c_info["fig"], use_container_width=True)
        else:
            st.info("No automatic charts could be generated. Use the custom chart builder below.")
            
        # Custom Chart Builder
        st.markdown('<div class="section-header">🛠️ Custom Chart Builder</div>', unsafe_allow_html=True)
        df_clean = bundle["df"]
        
        all_cols = list(df_clean.columns)
        
        col_c1, col_c2, col_c3, col_c4 = st.columns(4)
        with col_c1:
            custom_type = st.selectbox("Chart Type", ["bar", "line", "scatter", "pie", "box", "histogram", "heatmap"])
        with col_c2:
            custom_x = st.selectbox("X Axis (or Feature)", all_cols)
        with col_c3:
            custom_y = st.selectbox("Y Axis (Metric)", [None] + all_cols, index=0)
        with col_c4:
            custom_color = st.selectbox("Color Segment Column (Optional)", [None] + all_cols, index=0)
            
        # Auto-render the custom chart dynamically as selections change (prevents disappearing charts and button-state resets)
        if custom_type not in ["histogram", "heatmap"] and custom_y is None:
            st.info("💡 Please select a **Y Axis (Metric)** column to render the custom chart.")
        else:
            with st.spinner("Plotting custom layout..."):
                try:
                    master_agent = MasterAgent()
                    fig = master_agent.viz_agent.build_custom_chart(
                        df_clean,
                        chart_type=custom_type,
                        x_col=custom_x,
                        y_col=custom_y,
                        color_col=custom_color
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not build chart: {e}. Check that variables are compatible (e.g. numerical Y for line/scatter).")
                    
    # --- TAB 4: AI BUSINESS INSIGHTS ---
    with tab_insights:
        st.markdown('<div class="section-header">AI Business Narrative Reports</div>', unsafe_allow_html=True)
        
        insights = bundle["insights"]
        
        st.markdown("### Executive Summary")
        st.write(insights["executive_summary"])
        st.markdown("---")
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("### Key Strategic Findings")
            for finding in insights["key_findings"]:
                st.markdown(f"• {finding}")
                
        with col_f2:
            st.markdown("### Trend & Anomaly Diagnostics")
            for trend in insights["trends_anomalies"]:
                st.markdown(f"• {trend}")
                
        st.markdown("---")
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.markdown("### Growth & Optimization Opportunities")
            for opp in insights["opportunities"]:
                st.markdown(f"• {opp}")
                
        with col_o2:
            st.markdown("### Identified Risks & Pain Points")
            for risk in insights["risks"]:
                st.markdown(f"• {risk}")
                
    # --- TAB 5: STRATEGIC RECOMMENDATIONS ---
    with tab_recs:
        st.markdown('<div class="section-header">Actionable Business Recommendations</div>', unsafe_allow_html=True)
        
        recs = bundle["recommendations"]
        
        for r in recs:
            render_recommendation_card(
                title=r["title"],
                description=r["description"],
                priority=r.get("priority", "Medium"),
                impact=r.get("impact", "Medium Impact"),
                confidence=r.get("confidence", 85)
            )
            
    # --- TAB 6: AI CONSULTANT CHAT ---
    with tab_chat:

        st.markdown(
            '<div class="section-header">💬 Ask Questions About Your Data</div>',
            unsafe_allow_html=True
        )

        # Initialize chat history and summary states
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        if "chat_summary" not in st.session_state:
            st.session_state["chat_summary"] = None

        if "last_summary_length" not in st.session_state:
            st.session_state["last_summary_length"] = 0

        # Create a container for messages so everything rendered inside appears ABOVE the chat input box
        chat_container = st.container()

        # Display previous messages in the container
        with chat_container:
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat input at bottom
        user_question = st.chat_input(
            "Ask anything about your dataset, e.g. 'Which brand has highest revenue?'"
        )

        if user_question:

            # Show user message immediately inside the container (above the input bar!)
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_question)

            # Append user message to history
            st.session_state["chat_history"].append({
                "role": "user",
                "content": user_question
            })

            # Check if chat history exceeds 20 messages. If so, summarize.
            if len(st.session_state["chat_history"]) > 20:
                last_len = st.session_state.get("last_summary_length", 0)
                if st.session_state["chat_summary"] is None or (len(st.session_state["chat_history"]) - last_len) >= 6:
                    from utils.config import get_llm
                    with chat_container:
                        with st.spinner("Summarizing chat history for memory optimization..."):
                            st.session_state["chat_summary"] = summarize_chat_history(
                                st.session_state["chat_history"],
                                get_llm()
                            )
                            st.session_state["last_summary_length"] = len(st.session_state["chat_history"])

            # Generate assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing dataset..."):
                        # Run the ChatAgent with capped history, cached context, and optional summary
                        ans = bundle["chat_agent"].run(
                            user_question,
                            st.session_state["chat_history"],
                            chat_summary=st.session_state.get("chat_summary")
                        )
                        st.markdown(ans)

            # Save assistant response
            st.session_state["chat_history"].append({
                "role": "assistant",
                "content": ans
            })
            
            # Rerun so that history loops draw everything in clean order on next paint
            st.rerun()
        
    # --- TAB 7: REPORT EXPORTS ---
    with tab_export:
        st.markdown('<div class="section-header">Download Executive Reports</div>', unsafe_allow_html=True)
        st.write("Export a formal report summarizing the KPIs, key findings, and recommendations. Ideal for distribution to C-level executives and stakeholders.")
        
        col_ex1, col_ex2 = st.columns(2)
        
        with col_ex1:
            st.subheader("PDF Document (.pdf)")
            st.write("Generates a print-ready, styled PDF report including tables and formatted sections.")
            if st.button("Generate PDF Report", type="primary"):
                with st.spinner("Compiling PDF flowables..."):
                    try:
                        pdf_data = generate_pdf_report(bundle)
                        st.download_button(
                            label="📥 Download PDF",
                            data=pdf_data,
                            file_name="executive_business_report.pdf",
                            mime="application/pdf"
                        )
                        st.success("PDF generated successfully!")
                    except Exception as e:
                        st.error(f"Error compiling PDF: {e}")
                        
        with col_ex2:
            st.subheader("Microsoft Word Document (.docx)")
            st.write("Generates an editable DOCX report with standard heading hierarchies and styling.")
            if st.button("Generate Word Report", type="primary"):
                with st.spinner("Compiling Word document..."):
                    try:
                        docx_data = generate_docx_report(bundle)
                        st.download_button(
                            label="📥 Download Word Doc",
                            data=docx_data,
                            file_name="executive_business_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        st.success("DOCX generated successfully!")
                    except Exception as e:
                        st.error(f"Error compiling DOCX: {e}")

else:
    # 6. Welcome Display (when no file uploaded)
    st.info("👋 Welcome! Please upload a dataset file in the upload zone above to begin the automated business analysis.")
    
    st.markdown('<div class="section-header">💡 Try Uploading Sample Data Structures</div>', unsafe_allow_html=True)
    st.write(
        "You can upload standard transactional sheets, including:\n"
        "- **Sales Records**: columns like Date, Sales/Revenue, Quantity, Product, Customer ID, Cost.\n"
        "- **Marketing Metrics**: columns like Campaign Name, Clicks, Impressions, Conversions, Spend, Revenue.\n"
        "- **HR Data**: columns like Employee ID, Department, Salary, Performance Score, Attrition Status.\n"
        "- **Financial Ledgers**: columns like Date, Revenue, Expenses, Net Profit, Assets, Liabilities."
    )
