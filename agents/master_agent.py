import pandas as pd
from agents.base_agent import BaseAgent
from agents.data_agent import DataAgent
from agents.kpi_agent import KPIAgent
from agents.dataset_intelligence_agent import DatasetIntelligenceAgent
from agents.visualization_agent import VisualizationAgent
from agents.insight_agent import InsightAgent
from agents.recommendation_agent import RecommendationAgent
from agents.chat_agent import ChatAgent
from rag.retrieval import index_analysis_outputs

class MasterAgent(BaseAgent):
    """
    Master Business Analyst Agent: Coordinates the entire analytics pipeline
    by orchestrating the specialized sub-agents.
    """
    def __init__(self):
        super().__init__(
            name="Senior Business Analyst",
            role="Lead orchestrator coordinating data, KPI, insight, visualization, and strategic recommendation pipelines."
        )
        self.data_agent = DataAgent()
        self.kpi_agent = KPIAgent()
        self.intelligence_agent = DatasetIntelligenceAgent()
        self.viz_agent = VisualizationAgent()
        self.insight_agent = InsightAgent()
        self.rec_agent = RecommendationAgent()
        self.chat_agent = ChatAgent()

    def run(self, df: pd.DataFrame, clean_data: bool = False, fill_missing: bool = True,
            remove_duplicates: bool = True, handle_outliers: bool = False,
            hidden_columns: list = None) -> dict:
        """
        Orchestrates the analysis pipeline.
        Returns:
            dict containing all calculations, recommendations, visualizations, and chat context.
        """
        # Step 1: Profile and (optionally) clean dataset
        data_results = self.data_agent.run(
            df, 
            clean_data=clean_data, 
            fill_missing=fill_missing, 
            remove_duplicates=remove_duplicates, 
            handle_outliers=handle_outliers
        )
        cleaned_df = data_results["cleaned_df"]
        overview = data_results["overview"]
        quality_report = data_results["quality_report"]
        cleaning_logs = data_results["cleaning_logs"]
        
        # Create a filtered dataframe without hidden columns for KPI/Insight/Viz calculations
        analysis_df = cleaned_df.copy()
        if hidden_columns:
            analysis_df.drop(columns=hidden_columns, inplace=True, errors="ignore")
            
        # Step 2: Auto-detect theme & dynamic KPIs using Dataset Intelligence
        intel_results = self.intelligence_agent.run(analysis_df)
        intel_domain = intel_results["domain"]
        intel_theme = intel_results["theme"]
        intel_kpis = intel_results["kpis"]
        suggested_charts = intel_results["suggested_charts"]
        
        # Step 3: Run traditional domain detection for backward compatibility
        kpi_results = self.kpi_agent.run(analysis_df)
        trad_domain = kpi_results["domain"]
        trad_col_map = kpi_results["mapped_columns"]
        trad_kpis = kpi_results["kpis"]
        
        # Decide domain & merge KPIs
        if trad_domain != "Generic":
            domain = trad_domain
            col_map = trad_col_map
            # Merge: dynamic KPIs + traditional KPIs (traditional wins if duplicate keys to ensure reports compatibility)
            kpis = {**intel_kpis, **trad_kpis}
        else:
            domain = intel_theme
            col_map = trad_col_map
            kpis = intel_kpis
            
        # Step 4: Recommend & Build Visualizations
        recommended_charts = self.viz_agent.run(cleaned_df, domain, col_map, suggested_charts=suggested_charts)
        
        # Step 5: Generate Executive Insights via LLM
        insights = self.insight_agent.run(overview, kpis, domain)
        
        # Step 6: Generate Business Recommendations via LLM
        recommendations = self.rec_agent.run(domain, kpis, insights)
        
        # Step 7: Index reports in the RAG Document Store
        doc_store = index_analysis_outputs(overview, kpis, insights, recommendations)
        
        # Step 8: Configure Chat Agent Context with full dataframe and hidden column info
        self.chat_agent.set_context(cleaned_df, doc_store, hidden_columns=hidden_columns)
        
        return {
            "df": cleaned_df,
            "domain": domain,
            "col_map": col_map,
            "overview": overview,
            "quality_report": quality_report,
            "cleaning_logs": cleaning_logs,
            "kpis": kpis,
            "charts": recommended_charts,
            "insights": insights,
            "recommendations": recommendations,
            "doc_store": doc_store,
            "chat_agent": self.chat_agent,
            "hidden_columns": hidden_columns or []
        }

