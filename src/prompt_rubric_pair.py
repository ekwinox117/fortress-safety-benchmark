from dataclasses import dataclass
from typing import List, Optional

@dataclass
class PromptRubricPair:
    """Represents a prompt and its associated rubric for evaluation."""
    id: int
    risk_domain: str
    risk_subdomain: str
    adversarial_prompt: str
    benign_prompt: str
    rubric_questions: List[str]
    additional_notes: Optional[str] = None