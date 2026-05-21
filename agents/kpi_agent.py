import pandas as pd
from agents.base_agent import BaseAgent
from analytics.kpi_engine import detect_domain_and_columns, compute_kpis

class KPIAgent(BaseAgent):
    """
    KPI Agent: Identifies relevant KPI concepts based on columns
    and calculates operational and financial business metrics.
    """
    def __init__(self):
        super().__init__(
            name="KPI Detection Agent",
            role="Specialist in identifying business domains, mapping metrics, and calculating operational/financial KPIs."
        )

    def run(self, df: pd.DataFrame) -> dict:
        """
        Executes KPI detection and calculations on the dataset.
        Returns:
            dict containing:
                - domain: Sales, Marketing, HR, Finance, or Generic
                - mapped_columns: dictionary of conceptual column mapping
                - kpis: dictionary of computed KPIs and format guides
        """
        domain, col_map = detect_domain_and_columns(df)
        kpis = compute_kpis(df, domain, col_map)
        
        return {
            "domain": domain,
            "mapped_columns": col_map,
            "kpis": kpis
        }
