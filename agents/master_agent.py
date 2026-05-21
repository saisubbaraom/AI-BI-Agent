import pandas as pd
from agents.base_agent import BaseAgent
from agents.data_agent import DataAgent
from agents.kpi_agent import KPIAgent
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
        self.viz_agent = VisualizationAgent()
        self.insight_agent = InsightAgent()
        self.rec_agent = RecommendationAgent()
        self.chat_agent = ChatAgent()

    def run(self, df: pd.DataFrame, clean_data: bool = False, fill_missing: bool = True,
            remove_duplicates: bool = True, handle_outliers: bool = False) -> dict:
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
        
        # Step 2: Auto-detect KPIs
        kpi_results = self.kpi_agent.run(cleaned_df)
        domain = kpi_results["domain"]
        col_map = kpi_results["mapped_columns"]
        kpis = kpi_results["kpis"]
        
        # Step 3: Recommend Visualizations
        recommended_charts = self.viz_agent.run(cleaned_df, domain, col_map)
        
        # Step 4: Generate Executive Insights via Grok LLM
        insights = self.insight_agent.run(overview, kpis, domain)
        
        # Step 5: Generate Business Recommendations via Grok LLM
        recommendations = self.rec_agent.run(domain, kpis, insights)
        
        # Step 6: Index reports in the RAG Document Store
        doc_store = index_analysis_outputs(overview, kpis, insights, recommendations)
        
        # Step 7: Configure Chat Agent Context
        self.chat_agent.set_context(cleaned_df, doc_store)
        
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
            "chat_agent": self.chat_agent
        }
