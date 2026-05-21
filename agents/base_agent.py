from abc import ABC, abstractmethod
from utils.config import get_llm

class BaseAgent(ABC):
    """
    Abstract base class for all specialized business analytics agents.
    """
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        
    @property
    def llm(self):
        """Get the LLM dynamically based on current configuration to prevent caching outdated settings."""
        return get_llm()

    @abstractmethod
    def run(self, *args, **kwargs):
        """Execute the agent's core responsibility."""
        pass
