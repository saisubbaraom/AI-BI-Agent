# Business Analyst AI Agent 📊🤖

An intelligent, multi-agent business analytics web application that acts as a professional Business Analyst and Strategy Consultant. Users upload raw data files (CSV, Excel, or Parquet), and the platform automatically profiles data quality, identifies the domain, calculates key metrics, creates Plotly charts, writes executive reports via Grok LLM (xAI), and supports conversational analytics using a hybrid RAG chat.

---

## 🏗️ Multi-Agent Architecture

The system coordinates specialized agents to analyze your datasets:
1. **Data Profiling Agent**: Audits data structures, reports quality stats, flags missing records, detects outliers, and isolates customer demographic categories.
2. **KPI Detection Agent**: Detects the business domain (Sales, Marketing, HR, or Finance) and calculates domain-specific metrics.
3. **Visualization Agent**: Evaluates variables to generate standard, interactive Plotly dashboard charts, and supports custom plots.
4. **Business Insights Agent**: Narrates the data trends, summary observations, and operational bottlenecks by querying the xAI Grok LLM.
5. **Recommendation Agent**: Formulates high-priority action plans detailing priority rankings, expected impact, and confidence scores.
6. **Chat Agent**: Drives conversational data consultancies using an advanced hybrid RAG retriever over statistical tables and reports.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- An xAI API Key (for Grok models)

### 1. Install Dependencies
Initialize a terminal inside the project directory and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Open the `.env` file in the root directory and add your xAI credentials:
```env
XAI_API_KEY=your_grok_api_key_here
```
*Note: You can also enter your API key directly inside the Streamlit sidebar.*

### 3. Start the Web App
Run the Streamlit server to open the dashboard in your default browser:
```bash
streamlit run app.py
```

---

## 💡 Supported Dataset Domains

The KPI detection engine maps concepts based on columns, including:
- **Sales / E-Commerce**: Calculates Total Revenue, Average Order Value (AOV), Profit Margin, Average Value per Customer (CLV), and Units Sold.
- **Marketing / Advertising**: Calculates Spend, Clicks, Impressions, CPA, CVR, CTR, CPC, and ROAS.
- **HR / Talent Management**: Calculates Headcount, Attrition Rate, Average Salary, and Average Performance Score.
- **Finance / Ledgers**: Calculates Total Revenue, Expenses, Net Profit Margin, ROI, and Asset/Liability ratio.
- **Generic Tabular**: Computes descriptive statistics (sums, means, medians, ranges) grouped by category.
