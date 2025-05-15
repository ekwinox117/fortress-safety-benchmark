from abc import ABC, abstractmethod

class ModelProvider(ABC):
    """Base class for all model providers."""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        pass 