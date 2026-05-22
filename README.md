# Business Analyst AI Agent 📊🤖

An intelligent, multi-agent business analytics web application that acts as a professional Business Analyst and Strategy Consultant. Users upload raw data files (CSV, Excel, or Parquet), and the platform automatically profiles data quality, identifies the domain/theme dynamically, calculates key metrics, creates Plotly charts, writes executive reports via Grok LLM (xAI), and supports conversational analytics using a hybrid RAG chat.

---

## 🏗️ Core Architecture & Features

The system coordinates specialized agents to analyze your datasets:
1. **Dataset Intelligence Agent**: Profiles schemas, runs heuristics to auto-flag PII & IDs, infers custom dataset themes (e.g. EV adoption, patient healthcare), suggests custom business KPIs with Pandas expressions, and designs visualization configurations.
2. **Relationship & Merge Planner**: Scans multiple datasets, suggests joins, calculates key overlap coverage, flags duplicate warnings, shows estimated row counts, and renders an interactive **Mermaid relationship graph** showing table connections.
3. **Data Preprocessing Column Wizard**: Lets users choose how to treat columns prior to running analysis: **Keep** (full use), **Hide** (remove from front-end charts/KPIs but retain in database for AI consultant Q&A), or **Exclude** (drop completely from dataset).
4. **Business Insights & Strategy Agents**: Synthesizes narrative observations, trends, market opportunities, and operational risks, providing ranked operational recommendation checklists with impact/confidence scores.
5. **AI Consultant Chat**: Supports full conversational analysis over your data using a hybrid retrieval mechanism (combining semantic document search over insights with a structured **Pandas Query Engine** that translates questions into secure Python aggregations).

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

### 3. Run Verification Tests
Verify the pipeline works correctly with standard and universal datasets:
```bash
python -m scratch.test_universal_pipeline
```

### 4. Start the Web App
Run the Streamlit server to open the dashboard in your default browser:
```bash
streamlit run app.py
```

---

## 📖 Comprehensive Documentation
Detailed explanations of the architecture, file layout, data lifecycle flows, and mathematical matching formulas are available in the [documentation.md](file:///C:/Users/saisu/.gemini/antigravity-ide/brain/f1460ded-9838-43a9-b192-3b9f5dd1c06c/documentation.md) artifact.
