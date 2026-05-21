import json
from agents.base_agent import BaseAgent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser

class RecommendationAgent(BaseAgent):
    """
    Recommendation Agent: Formulates concrete, prioritized, and high-impact
    business action items based on KPIs and narrative findings.
    """
    def __init__(self):
        super().__init__(
            name="Recommendation Agent",
            role="Specialist in translating analytical findings into structured strategic action plans."
        )

    def run(self, domain: str, kpis: dict, insights: dict) -> list[dict]:
        """
        Invokes Grok LLM to generate 4-6 prioritized recommendations.
        """
        # Format KPIs
        kpis_str = "\n".join([f"- {info['label']}: {info['value']}" for info in kpis.values()])
        
        # Format insights
        findings_str = "\n".join([f"- {item}" for item in insights.get("key_findings", [])])
        opportunities_str = "\n".join([f"- {item}" for item in insights.get("opportunities", [])])
        risks_str = "\n".join([f"- {item}" for item in insights.get("risks", [])])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite Business Strategy Advisor. "
                "Based on the provided KPIs and narrative insights, generate 4-6 highly specific, "
                "actionable, and realistic business recommendations.\n\n"
                "You MUST return your response as a valid JSON array of objects. "
                "Do not include any wrapper markdown or surrounding text, just the raw JSON array. "
                "Each object in the array must follow this exact structure:\n"
                "[\n"
                "  {{\n"
                '    "title": "Short title of recommendation (e.g., Optimize Customer Acquisition)",\n'
                '    "description": "Detailed explanation of the recommendation, what actions to take, and why it is suggested by the data.",\n'
                '    "priority": "Critical" | "High" | "Medium" | "Low",\n'
                '    "impact": "High Impact" | "Medium Impact" | "Low Impact",\n'
                '    "confidence": 85\n'
                "  }},\n"
                "  ...\n"
                "]\n"
                "Keep the JSON syntax clean. Double-check that priority and impact match the allowed values exactly."
            )),
            ("user", (
                "Domain: {domain}\n\n"
                "Calculated KPIs:\n{kpis}\n\n"
                "Core Findings:\n{findings}\n\n"
                "Growth Opportunities:\n{opportunities}\n\n"
                "Risks Identified:\n{risks}\n"
            ))
        ])
        
        try:
            chain = prompt | self.llm | SimpleJsonOutputParser()
            inputs = {
                "domain": domain,
                "kpis": kpis_str,
                "findings": findings_str,
                "opportunities": opportunities_str,
                "risks": risks_str
            }
            return chain.invoke(inputs)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return self._get_fallback_recommendations(domain)

    def _get_fallback_recommendations(self, domain: str) -> list[dict]:
        """Provides static high-value business recommendations if Grok is offline."""
        if domain == "Sales":
            return [
                {
                    "title": "Implement Re-engagement Campaigns for High-Value Segments",
                    "description": "Analyze purchase intervals and target lapsed customers who contributed to top-tier revenue segments. Introduce tailored loyalty promotions.",
                    "priority": "High",
                    "impact": "High Impact",
                    "confidence": 90
                },
                {
                    "title": "Optimize Average Order Value (AOV) via Bundle Offers",
                    "description": "Establish automated product bundling on the checkout page by matching complementary items. Offer a small discount (e.g., 5-10%) on bundles.",
                    "priority": "Medium",
                    "impact": "Medium Impact",
                    "confidence": 85
                },
                {
                    "title": "Audit Cost of Goods Sold (COGS) to Expand Margin",
                    "description": "Review supplier terms for products with high sales volume but narrow margins. Negotiate bulk pricing or look for alternative distributors.",
                    "priority": "Critical",
                    "impact": "High Impact",
                    "confidence": 95
                }
            ]
        elif domain == "Marketing":
            return [
                {
                    "title": "Reallocate Budget from High-CPA to High-ROAS Campaigns",
                    "description": "Audit channel-specific performance. Shift ad spend away from campaigns showing rising CPC/CPA metrics and double-down on proven conversion channels.",
                    "priority": "Critical",
                    "impact": "High Impact",
                    "confidence": 95
                },
                {
                    "title": "Perform A/B Testing on Low Click-Through Rate (CTR) Ad Creatives",
                    "description": "Deploy fresh copywriting and color-contrast designs for campaigns with high impressions but low click rates. Run experiments for 14 days.",
                    "priority": "High",
                    "impact": "Medium Impact",
                    "confidence": 80
                },
                {
                    "title": "Enhance Post-Click Conversion Rate (CVR) via Landing Page Optimization",
                    "description": "Align landing page headers directly with campaign hooks, speed up page load speeds, and simplify form fields to reduce visitor drop-off.",
                    "priority": "High",
                    "impact": "High Impact",
                    "confidence": 90
                }
            ]
        elif domain == "HR":
            return [
                {
                    "title": "Conduct Structured Stay Interviews in High-Attrition Departments",
                    "description": "Gather qualitative feedback from employees in departments exhibiting attrition rates higher than the corporate average. Focus on workload and management style.",
                    "priority": "Critical",
                    "impact": "High Impact",
                    "confidence": 90
                },
                {
                    "title": "Align Compensation Structure with Market Baselines",
                    "description": "Evaluate employee salaries relative to industry standards, especially for technical or specialized roles. Adjust packages to decrease recruitment costs.",
                    "priority": "High",
                    "impact": "High Impact",
                    "confidence": 95
                },
                {
                    "title": "Design Career Progression Roadmaps",
                    "description": "Establish clear qualification criteria for internal promotions. Provide employee training stipends to boost skill development and long-term engagement.",
                    "priority": "Medium",
                    "impact": "Medium Impact",
                    "confidence": 85
                }
            ]
        else:
            return [
                {
                    "title": "Establish Regular Data Hygiene Audits",
                    "description": "Clean duplicate records and standardize input schema layouts across data-entry systems. This guarantees higher statistical accuracy in future dashboards.",
                    "priority": "High",
                    "impact": "Medium Impact",
                    "confidence": 90
                },
                {
                    "title": "Set Up Automated Alerting on Outlier Thresholds",
                    "description": "Configure metric monitors that notify business stakeholders when critical aggregates deviate from standard historical distributions.",
                    "priority": "Medium",
                    "impact": "Medium Impact",
                    "confidence": 80
                }
            ]
        
