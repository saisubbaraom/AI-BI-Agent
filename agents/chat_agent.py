from agents.base_agent import BaseAgent
from rag.retrieval import StructuredPandasQueryEngine
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class ChatAgent(BaseAgent):
    """
    Chat Agent: Handles interactive Q&A over the dataset, insights, KPIs,
    and recommendations. Uses a hybrid retrieval approach.
    """
    def __init__(self, df=None, doc_store=None):
        super().__init__(
            name="AI Business Consultant",
            role="Specialist in answering operational, technical, and strategic questions about business data."
        )
        self.df = df
        self.doc_store = doc_store
        self.query_engine = None

    def set_context(self, df, doc_store):
        """Update dataset and document store context."""
        self.df = df
        self.doc_store = doc_store
        if df is not None:
            self.query_engine = StructuredPandasQueryEngine(df, self.llm)

    def run(self, question: str, chat_history: list, chat_summary: str = None) -> str:
        """
        Processes a user question, retrieves relevant context,
        and generates a professional business answer via Groq LLM.
        """
        # 1. Retrieve unstructured document chunks (schema, findings, recommendations)
        retrieved_docs = []
        if self.doc_store:
            retrieved_docs = self.doc_store.query(question)

        docs_context = "\n\n".join([
            f"[Source: {doc.metadata.get('type', 'Report')}]:\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        # 2. Run structured Pandas query if dataset is available
        query_context = ""
        if self.query_engine:
            self.query_engine.llm = self.llm
            plan = self.query_engine.generate_query_plan(question)
            if plan and plan.get("query_needed", False):
                query_context = self.query_engine.execute_query_plan(plan)

        # 3. Build messages list including chat history
        system_content = (
            "You are an elite Business Consultant Chat Agent. "
            "You have access to a dataset's metadata, computed KPIs, business insights, "
            "and a structured query runner. Answer the user's question about the data in "
            "a highly professional, concise, and structured manner.\n\n"
        )
        if chat_summary:
            system_content += f"=== CONVERSATION SUMMARY SO FAR ===\n{chat_summary}\n\n"

        system_content += (
            "When answering:\n"
            "- State calculations and figures clearly.\n"
            "- Connect raw calculations to business impacts (e.g. 'This means that...').\n"
            "- If the user asks for charts, tell them to use the 'Visualizations' tab.\n"
            "- Base your answers strictly on the retrieved context. Do not invent details.\n\n"
            f"=== DOCUMENT SEARCH CONTEXT ===\n{docs_context}\n\n"
            f"=== STRUCTURED QUERY OUTPUT ===\n{query_context}\n"
        )

        messages = [SystemMessage(content=system_content)]

        # Add chat history (up to last 5 messages)
        for msg in chat_history[-5:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # Add latest question
        messages.append(HumanMessage(content=question))

        try:
            response = self.llm.invoke(messages)
            return response.content
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate_limit" in err_msg.lower() or "rate limit" in err_msg.lower():
                return (
                    "⚠️ **Rate Limit Exceeded (429)**: The AI service is currently receiving too many requests. "
                    "Please wait a moment before sending another question or try refreshing/switching the model in settings."
                )
            return f"Error communicating with AI: {err_msg}"
