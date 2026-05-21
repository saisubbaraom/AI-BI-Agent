import pandas as pd
from agents.base_agent import BaseAgent
from analytics.profiling import get_dataset_overview, get_data_quality_report, analyze_demographics
from utils.data_cleaner import clean_dataset

class DataAgent(BaseAgent):
    """
    Data Agent: Specializes in data profiling, cleaning, quality audits,
    and automatic demographics detection.
    """
    def __init__(self):
        super().__init__(
            name="Data Profiling Agent",
            role="Expert in data auditing, cleaning, outlier detection, and demographic profiling."
        )

    def run(self, df: pd.DataFrame, clean_data: bool = False, fill_missing: bool = True, 
            remove_duplicates: bool = True, handle_outliers: bool = False) -> dict:
        """
        Runs the profiling pipeline. Optionally cleans the dataset.
        Returns:
            dict containing overview, quality report, demographics, and cleaning logs.
        """
        # If clean is requested, process the dataframe first
        cleaning_logs = None
        cleaned_df = df
        
        if clean_data:
            cleaned_df, cleaning_logs = clean_dataset(
                df, 
                fill_missing=fill_missing, 
                remove_duplicates=remove_duplicates, 
                handle_outliers=handle_outliers
            )
            
        overview = get_dataset_overview(cleaned_df)
        quality = get_data_quality_report(cleaned_df)
        demographics = analyze_demographics(cleaned_df)
        
        # Inject demographics analysis back into overview metadata
        overview["demographics"] = demographics
        
        return {
            "cleaned_df": cleaned_df,
            "overview": overview,
            "quality_report": quality,
            "cleaning_logs": cleaning_logs
        }
