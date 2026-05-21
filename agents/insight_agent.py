from agents.base_agent import BaseAgent
from analytics.insights import generate_business_insights

class InsightAgent(BaseAgent):
    """
    Insight Agent: Discovers strategic narratives, identifies trends,
    highlights data anomalies, and builds executive findings using Grok.
    """
    def __init__(self):
        super().__init__(
            name="Business Insights Agent",
            role="Specialist in interpreting raw statistical calculations into high-level business narratives."
        )

    def run(self, dataset_info: dict, kpis: dict, domain: str) -> dict:
        """
        Executes Grok LLM analysis.
        Returns:
            dict containing executive_summary, key_findings, trends_anomalies, opportunities, and risks.
        """
        if not self.llm:
            # Fallback to local prompt-less generation if LLM is not configured
            # (which is handled inside generate_business_insights)
            pass
            
        return generate_business_insights(dataset_info, kpis, domain, self.llm)
