import os
import unittest
from unittest.mock import MagicMock, patch
from utils.config import get_llm
from utils.helpers import summarize_chat_history
from agents.chat_agent import ChatAgent
from rag.retrieval import StructuredPandasQueryEngine
import pandas as pd

class TestChatRefactor(unittest.TestCase):
    def test_fallback_chain(self):
        print("Testing fallback chain structure...")
        with patch("utils.config.get_model_name", return_value="llama-3.3-70b-versatile"):
            with patch.dict(os.environ, {"GEMINI_API_KEY": "fake_gemini_key"}):
                llm = get_llm()
                self.assertTrue(hasattr(llm, "runnable") or hasattr(llm, "fallbacks"))
                primary = getattr(llm, "runnable", llm)
                
                # Check model name on primary
                primary_model = getattr(primary, "model_name", getattr(primary, "model", None))
                self.assertEqual(primary_model, "llama-3.3-70b-versatile")
                
                fallbacks = getattr(llm, "fallbacks", [])
                self.assertTrue(len(fallbacks) >= 2, f"Expected at least 2 fallbacks, got {len(fallbacks)}")
                
                fb0_model = getattr(fallbacks[0], "model_name", getattr(fallbacks[0], "model", None))
                fb1_model = getattr(fallbacks[1], "model_name", getattr(fallbacks[1], "model", None))
                
                self.assertEqual(fb0_model, "llama-3.1-8b-instant")
                self.assertEqual(fb1_model, "gemini-2.5-flash")

    def test_summarize_chat_history(self):
        print("Testing chat history summarization...")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Summary of conversation: user asked about revenue."
        mock_llm.invoke.return_value = mock_response
        
        chat_history = [
            {"role": "user", "content": "hello, who are you?"},
            {"role": "assistant", "content": "I am your business analyst AI."},
            {"role": "user", "content": "what is the highest selling brand?"}
        ]
        
        summary = summarize_chat_history(chat_history, mock_llm)
        self.assertEqual(summary, "Summary of conversation: user asked about revenue.")
        mock_llm.invoke.assert_called_once()

    def test_query_preview_limit(self):
        print("Testing StructuredPandasQueryEngine preview cap limit...")
        df = pd.DataFrame({"A": range(20)})
        mock_llm = MagicMock()
        engine = StructuredPandasQueryEngine(df, mock_llm)
        
        plan = {
            "query_needed": True,
            "query_type": "filter_only",
            "groupby_column": "A",
            "limit": 15
        }
        res = engine.execute_query_plan(plan)
        self.assertIn("Top 10", res)
        
        plan["limit"] = 5
        res = engine.execute_query_plan(plan)
        self.assertIn("Top 5", res)

    def test_chat_agent_run_history_cap(self):
        print("Testing ChatAgent limits messages to last 5...")
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "ChatAgent reply."
        mock_llm.invoke.return_value = mock_response
        
        chat_history = [
            {"role": "user", "content": f"msg {i}"} if i % 2 == 0 else {"role": "assistant", "content": f"reply {i}"}
            for i in range(20)
        ]
        
        agent = ChatAgent()
        with patch.object(ChatAgent, "llm", mock_llm):
            agent.run("latest question", chat_history)
            invoked_messages = mock_llm.invoke.call_args[0][0]
            self.assertEqual(len(invoked_messages), 7)
            self.assertEqual(invoked_messages[1].content, "reply 15")

if __name__ == "__main__":
    unittest.main()
