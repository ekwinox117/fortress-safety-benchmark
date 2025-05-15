import json
from typing import List, Dict, Optional
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from .prompt_rubric_pair import PromptRubricPair
from .models.base import ModelProvider

class DataLoader:
    def __init__(self, input_file: str):
        """Initialize the DataLoader with the path to the JSONL file."""
        self.input_file = input_file
        self.prompt_rubric_pairs: List[PromptRubricPair] = []
        self.adversarial_responses: Dict[int, str] = {}  # Store model response for each prompt
        self.benign_responses: Dict[int, str] = {}  # Store model response for each prompt

    def load_data(self):
        """Load prompt-rubric pairs from the JSONL file."""
        with open(self.input_file, 'r') as f:
            data = [json.loads(line) for line in f]
            
        self.prompt_rubric_pairs = []
        for row in data:
            # Create PromptRubricPair object
            # Convert ID to int if it exists, otherwise use a default ID
            id_value = row.get('ID')
            if id_value is not None:
                try:
                    id_value = int(id_value)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid ID format '{id_value}', using default 0")
                    id_value = 0
            else:
                id_value = 0
                
            pair = PromptRubricPair(
                id=id_value,
                risk_domain=row.get('risk_domain', ''),
                risk_subdomain=row.get('risk_subdomain', ''),
                adversarial_prompt=row.get('adversarial_prompt', ''),
                benign_prompt=row.get('benign_prompt', ''),
                rubric_questions=[question for question in row.get('rubric', [])],
                additional_notes=row.get('additional_notes', None)
            )
            self.prompt_rubric_pairs.append(pair)

    def get_prompt_rubric_pairs(self) -> List[PromptRubricPair]:
        """Returns the list of prompt-rubric pairs."""
        return self.prompt_rubric_pairs

    def get_responses(self, responses_type: str) -> Dict[int, str]:
        """Returns the dictionary of responses."""
        if responses_type == "adversarial":
            return self.adversarial_responses
        elif responses_type == "benign":
            return self.benign_responses

    def get_pair_by_id(self, pair_id: int) -> Optional[PromptRubricPair]:
        """Get a specific prompt-rubric pair by ID."""
        for pair in self.prompt_rubric_pairs:
            if pair.id == pair_id:
                return pair
        return None 
    
    def load_responses(self, file_path: str):
        """Load model responses from the provided file."""
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                # Convert ID to int for consistent type
                id_value = data.get('ID')
                if id_value is not None:
                    try:
                        id_value = int(id_value)
                    except (ValueError, TypeError):
                        print(f"Warning: Invalid ID format '{id_value}' in responses file, skipping")
                        continue
                    
                    self.adversarial_responses[id_value] = data['adversarial_response']
                    self.benign_responses[id_value] = data['benign_response']
                else:
                    print("Warning: Response data missing ID, skipping")
    
    def generate_adversarial_responses(self, model_provider: ModelProvider, max_workers: int = 1) -> Dict[int, str]:
        """Generate adversarial responses for each prompt-rubric pair using the provided model provider in parallel"""
        def generate_pair_response(pair: PromptRubricPair):
            try:
                response = model_provider.generate(
                    pair.adversarial_prompt,
                )
                return response
            except Exception as e:
                error_msg = f"Error generating response for pair {pair.id}: {str(e)}"
                print(f"{error_msg}. Exception saved as response.")
                return str(e)
        
        # Create a ThreadPoolExecutor to generate responses in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(generate_pair_response, pair): pair for pair in self.prompt_rubric_pairs}
            for future in tqdm(as_completed(future_to_pair), total=len(future_to_pair), desc="Generating adversarial responses"):
                pair = future_to_pair[future]
                try:
                    response = future.result()
                    self.adversarial_responses[pair.id] = response
                except Exception as e:
                    print(f"Error generating response for pair {pair.id}: {str(e)}")
                    self.adversarial_responses[pair.id] = str(e)

    def generate_benign_responses(self, model_provider: ModelProvider, max_workers: int = 1) -> Dict[int, str]:
        """Generate benign responses for each prompt-rubric pair using the provided model provider in parallel"""
        def generate_pair_response(pair: PromptRubricPair):
            try:
                response = model_provider.generate(
                    pair.benign_prompt,
                )
                return response
            except Exception as e:
                error_msg = f"Error generating response for pair {pair.id}: {str(e)}"
                print(f"{error_msg}. Exception saved as response.")
                return str(e)
        
        # Create a ThreadPoolExecutor to generate responses in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(generate_pair_response, pair): pair for pair in self.prompt_rubric_pairs}
            for future in tqdm(as_completed(future_to_pair), total=len(future_to_pair), desc="Generating benign responses"):
                pair = future_to_pair[future]
                try:
                    response = future.result()
                    self.benign_responses[pair.id] = response
                except Exception as e:
                    print(f"Error generating response for pair {pair.id}: {str(e)}")
                    self.benign_responses[pair.id] = str(e)
