import streamlit as st
import pandas as pd
import numpy as np

def format_number(val):
    """
    Format a number in a readable business format (e.g. $1.2M, $450K, 15.2%).
    """
    if pd.isna(val) or val is None:
        return "N/A"
    
    # Check if integer or float
    try:
        val = float(val)
    except (ValueError, TypeError):
        return str(val)
        
    abs_val = abs(val)
    
    if abs_val >= 1_000_000_000:
        return f"{val / 1_000_000_000:.2f}B"
    elif abs_val >= 1_000_000:
        return f"{val / 1_000_000:.2f}M"
    elif abs_val >= 1_000:
        return f"{val / 1_000:.2f}K"
    elif abs_val == 0:
        return "0.00"
    elif abs_val < 1:
        return f"{val:.4f}"
    else:
        return f"{val:.2f}"

def format_currency(val, currency_symbol="$"):
    """
    Format a value as currency.
    """
    formatted = format_number(val)
    if formatted == "N/A":
        return formatted
    if formatted.startswith("-"):
        return f"-{currency_symbol}{formatted[1:]}"
    return f"{currency_symbol}{formatted}"

def format_percent(val):
    """
    Format a value as a percentage. Expects decimal (e.g., 0.152 -> 15.2%)
    or already percentage (e.g., 15.2 -> 15.2% if value is > 1.0 or <-1.0).
    """
    if pd.isna(val) or val is None:
        return "N/A"
    try:
        val = float(val)
    except (ValueError, TypeError):
        return str(val)
        
    # If absolute value is small, assume it's decimal
    if abs(val) <= 1.0:
        return f"{val * 100:.2f}%"
    return f"{val:.2f}%"

def inject_premium_css():
    """
    Injects custom CSS to style the Streamlit interface for a premium, executive appearance.
    Includes glassmorphism cards, modern fonts, glow effects, and micro-animations.
    """
    st.markdown("""
        <style>
        /* Import Outfit and Inter Font families */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', -apple-system, sans-serif;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', -apple-system, sans-serif;
            font-weight: 600;
            letter-spacing: -0.02em;
        }
        
        /* Main title styling */
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FF6B6B 0%, #A855F7 50%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            text-shadow: 0px 10px 30px rgba(168, 85, 247, 0.1);
        }
        
        /* Subtitle */
        .sub-title {
            font-size: 1.1rem;
            color: #6B7280;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Glassmorphic Metric Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.15);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #A855F7, #3B82F6);
            opacity: 0.8;
        }
        
        /* Color themes for cards */
        .metric-card.sales::before { background: linear-gradient(90deg, #10B981, #059669); }
        .metric-card.marketing::before { background: linear-gradient(90deg, #F59E0B, #D97706); }
        .metric-card.hr::before { background: linear-gradient(90deg, #EC4899, #DB2777); }
        .metric-card.finance::before { background: linear-gradient(90deg, #3B82F6, #2563EB); }
        .metric-card.generic::before { background: linear-gradient(90deg, #6B7280, #4B5563); }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px 0 rgba(168, 85, 247, 0.2);
            border-color: rgba(168, 85, 247, 0.2);
        }
        
        .metric-title {
            font-size: 0.85rem;
            font-weight: 500;
            color: #9CA3AF;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            font-family: 'Outfit', sans-serif;
            color: #F3F4F6;
            margin-bottom: 0.2rem;
        }
        
        .metric-change {
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        .change-up { color: #10B981; }
        .change-down { color: #EF4444; }
        .change-neutral { color: #9CA3AF; }
        
        /* Styled sections */
        .section-header {
            border-left: 4px solid #A855F7;
            padding-left: 10px;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-weight: 600;
            color: #F3F4F6;
        }
        
        /* Tab styling */
        button[data-baseweb="tab"] {
            font-family: 'Outfit', sans-serif !important;
            font-weight: 500 !important;
            font-size: 1rem !important;
            transition: all 0.2s ease !important;
        }
        
        /* Recommendation Cards */
        .rec-card {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid rgba(255, 255, 255, 0.04);
            border-left: 5px solid #6B7280;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            transition: all 0.25s ease;
        }
        
        .rec-card.priority-critical { border-left-color: #EF4444; }
        .rec-card.priority-high { border-left-color: #F59E0B; }
        .rec-card.priority-medium { border-left-color: #3B82F6; }
        .rec-card.priority-low { border-left-color: #10B981; }
        
        .rec-card:hover {
            background: rgba(255, 255, 255, 0.04);
            transform: translateX(3px);
        }
        
        .rec-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #F3F4F6;
            margin-bottom: 0.4rem;
        }
        
        .rec-meta {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            font-size: 0.8rem;
            color: #9CA3AF;
            margin-bottom: 0.6rem;
        }
        
        .rec-tag {
            background: rgba(255, 255, 255, 0.05);
            padding: 2px 8px;
            border-radius: 4px;
            font-weight: 500;
        }
        
        .rec-desc {
            font-size: 0.95rem;
            color: #D1D5DB;
            line-height: 1.5;
        }
        
        /* Hide default Streamlit decoration */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        </style>
    """, unsafe_allow_html=True)

def render_kpi_card(title, value, change=None, change_direction="up", domain_class="generic"):
    """
    Renders a premium HTML/CSS KPI card in Streamlit.
    """
    change_html = ""
    if change is not None:
        if change_direction == "up":
            change_html = f'<div class="metric-change change-up">▲ {change}</div>'
        elif change_direction == "down":
            change_html = f'<div class="metric-change change-down">▼ {change}</div>'
        else:
            change_html = f'<div class="metric-change change-neutral">● {change}</div>'
            
    st.markdown(f"""
        <div class="metric-card {domain_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {change_html}
        </div>
    """, unsafe_allow_html=True)
    
def render_recommendation_card(title, description, priority="medium", impact="Medium Impact", confidence=85):
    """
    Renders a recommendation card with specific priority coloring.
    """
    priority_class = f"priority-{priority.lower()}"
    st.markdown(f"""
        <div class="rec-card {priority_class}">
            <div class="rec-title">{title}</div>
            <div class="rec-meta">
                <span class="rec-tag">Priority: {priority.upper()}</span>
                <span class="rec-tag">Impact: {impact}</span>
                <span class="rec-tag">Confidence: {confidence}%</span>
            </div>
            <div class="rec-desc">{description}</div>
        </div>
    """, unsafe_allow_html=True)

def summarize_chat_history(chat_history: list, llm) -> str:
    """
    Compiles the conversation log and prompts the LLM to write a concise 1-2 paragraph recap,
    summarizing key findings and topics discussed so far.
    """
    if not chat_history:
        return ""
    
    formatted_log = ""
    for msg in chat_history:
        role = "User" if msg.get("role") == "user" else "Assistant"
        formatted_log += f"{role}: {msg.get('content')}\n\n"
        
    prompt = (
        "You are an expert business analyst and technical summarizer.\n"
        "Provide a concise 1-2 paragraph summary of the key findings, data analysis questions, "
        "and insights discussed in the following chat conversation history. Focus only on the "
        "facts, queries, and results discussed. Do not add any greeting or generic conversation.\n\n"
        f"=== CONVERSATION LOG ===\n{formatted_log}\n\n"
        "Summary:"
    )
    
    from langchain_core.messages import HumanMessage
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error summarizing chat history: {e}")
        return "Failed to summarize chat history due to an API error."

