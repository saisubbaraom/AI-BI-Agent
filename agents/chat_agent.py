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
        self.hidden_columns = []
        self.query_engine = None

    def set_context(self, df, doc_store, hidden_columns=None):
        """Update dataset and document store context."""
        self.df = df
        self.doc_store = doc_store
        self.hidden_columns = hidden_columns or []
        if df is not None:
            self.query_engine = StructuredPandasQueryEngine(df, self.llm, hidden_columns=self.hidden_columns)

    def run(self, question: str, chat_history: list, chat_summary: str = None) -> str:
        """
        Processes a user question, retrieves relevant context,
        and generates a professional business answer via Groq LLM.
        """
        # Check if the question is a simple greeting/conversational phrase
        clean_q = "".join(c for c in question.lower() if c.isalnum() or c.isspace()).strip()
        greetings = {
            "hi", "hello", "hey", "greetings", "good morning", "good afternoon", 
            "good evening", "thanks", "thank you", "bye", "goodbye", "help", 
            "who are you", "what can you do", "what is your name"
        }
        
        is_greeting = clean_q in greetings or len(clean_q) <= 2
        
        # 1. Retrieve unstructured document chunks (schema, findings, recommendations)
        retrieved_docs = []
        if self.doc_store and not is_greeting:
            retrieved_docs = self.doc_store.query(question)

        docs_context = "\n\n".join([
            f"[Source: {doc.metadata.get('type', 'Report')}]:\n{doc.page_content}"
            for doc in retrieved_docs
        ])

        # 2. Run structured Pandas query if dataset is available
        query_context = ""
        if self.query_engine and not is_greeting:
            self.query_engine.llm = self.llm
            plan = self.query_engine.generate_query_plan(question, chat_history=chat_history)
            if plan and plan.get("query_needed", False):
                query_context = self.query_engine.execute_query_plan(plan)

        # 3. Build messages list including chat history
        if is_greeting:
            system_content = (
                "You are an elite Business Consultant Chat Agent. "
                "Respond to the user's greeting or simple conversational inquiry in a polite, helpful, and concise manner, "
                "letting them know you are ready to help analyze their business dataset."
            )
        else:
            system_content = (
                "You are an elite Business Consultant Chat Agent. "
                "You have access to a dataset's metadata, computed KPIs, business insights, "
                "and a structured query runner. Answer the user's question about the data in "
                "a highly professional, concise, and structured manner.\n\n"
            )
            if chat_summary:
                system_content += f"=== CONVERSATION SUMMARY SO FAR ===\n{chat_summary}\n\n"

            if self.hidden_columns:
                system_content += (
                    f"=== HIDDEN COLUMNS INFO ===\n"
                    f"The following columns are hidden from the user's dashboard view, but they still exist in the database and can be queried if the user asks: {', '.join(self.hidden_columns)}.\n"
                    "If the user asks questions about hidden columns, explain that they were hidden from the dashboard, but you are querying them anyway.\n\n"
                )

            system_content += (
                "When answering:\n"
                "- Do NOT be overly defensive about column names. If the user asks for 'brand', 'product', 'item', or 'category', and the database results group by 'product_category' (or similar column), treat the values in that column (e.g. 'Laptops', 'Smartphones') directly as the brands/products the user is asking about. Answer directly (e.g. 'Laptops had the highest sales...') instead of saying the data doesn't contain brand info.\n"
                "- State calculations and figures clearly. If you present profit margins or percentages, ensure they are mathematically correct. Do not mix overall totals with mean/average values. Note that Profit Margin is mathematically calculated as (Profit / Revenue). Never divide Revenue by Profit, and do not swap their labels (e.g. if Revenue is $100 and Profit is $30, the profit margin is 30% or 0.30, not 3.33).\n"
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
