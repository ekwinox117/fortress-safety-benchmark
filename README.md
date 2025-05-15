# Fortress Safety Benchmark

A tool for evaluating LLM responses to adversarial prompts using a standardized rubric.

## Project Structure

```
fortress-safety-benchmark/
├── data/                       # Data directory
│   ├── prompts_set.jsonl       # JSONL file containing prompt-rubric pairs
│   ├── test.jsonl              # Test data for development
│   └── sample_responses_file.jsonl # Sample model responses
├── src/                        # Source code
│   ├── config/                 # Configuration
│   │   └── models.py          # Model configuration
│   ├── models/                 # Model implementations
│   │   ├── base.py            # Base class for model providers
│   │   └── litellm.py         # LiteLLM model wrapper
│   ├── data_loader.py         # Data loading and processing
│   ├── evaluator.py           # Response evaluation logic
│   └── prompt_rubric_pair.py  # Prompt-rubric pair data structure
├── main.py                    # CLI entry point
└── README.md                  # This file
```

## Setup

1. Clone the repository
2. Add your data file to the `data/` directory:
   - The file should be a JSONL file containing prompt-rubric pairs. Sample files are provided!
3. Set up environment variables in a `.env` file:
   - API keys for the model providers you want to use

## Usage

```bash
python main.py --input-file data/prompts_set.jsonl --output-file results.json --model-args model=openai/gpt-4o
```

### Command-line Arguments

- `--input-file`: Path to the JSONL file containing prompts and rubrics (required)
- `--output-file`: Path to save the final evaluation results (required)
- `--responses-file`: Path to a JSONL file containing model responses (optional)
- `--model-args`: Model-specific arguments in key=value format (required if no responses file)
- `--max-workers-generate`: Number of parallel workers to use for generation (default: 1)
- `--max-workers-evaluate`: Number of parallel workers to use for evaluation (default: 1)
- `--raw-output`: Path to save raw output in CSV format (optional)

## Data Format

The input JSONL file should contain entries with the following fields:
- `ID`: Unique identifier for the prompt-rubric pair
- `risk_domain`: Risk domain category
- `risk_subdomain`: Risk subdomain category
- `adversarial_prompt`: The adversarial prompt text
- `benign_prompt`: The benign version of the prompt
- `rubric`: List of rubric questions for evaluation

## Evaluation Process

1. Load prompts from the JSONL file
2. Generate responses using the specified model (or load pre-generated responses)
3. Evaluate responses using the associated rubrics
4. Generate and save evaluation summary