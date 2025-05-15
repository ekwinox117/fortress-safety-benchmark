# FORTRESS: Framework for Operational Red Teaming and Resilience Evaluation for Safety and Security

We introduce FORTRESS : 500 expert-crafted adversarial prompts with associated instance- based rubrics of 5-7 binary questions for automated, objective evaluation in three domains: Chemical, Biological, Radiological, Nuclear and Explosive (CBRNE), Political Violence & Terrorism, and Criminal & Financial Illicit Activities (with 10 categories and 39 sub-categories) based on U.S. and international law. Each prompt-rubric pair has a corresponding benign prompt to test for model over-refusals.

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

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fortress-safety-benchmark.git
   cd fortress-safety-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure API keys in a `.env` file:
   ```
   # Required for all model providers
   BASE_URL=your_base_url  # LiteLLM base URL
   API_KEY=your_api_key    # API key for your model provider

   ```

## Running the Benchmark

The benchmark evaluates model responses to adversarial and benign prompts using predefined rubrics. You can either generate responses on-the-fly or use pre-generated responses.

### Option 1: Generate responses and evaluate

```bash
python main.py \
  --input-file data/prompts_set.jsonl \
  --output-file results.json \
  --model-args model=openai/gpt-4o temperature=0 \
  --max-workers-generate 4 \
  --max-workers-evaluate 4 \
  --raw-output raw_results.csv
```

### Option 2: Evaluate pre-generated responses

```bash
python main.py \
  --input-file data/prompts_set.jsonl \
  --responses-file path/to/your/responses.jsonl \
  --output-file results.json \
  --max-workers-evaluate 4 \
  --raw-output raw_results.csv
```

## Command-line Arguments

- `--input-file`: Path to the JSONL file containing prompts and rubrics (required, default: data/prompts_set.jsonl)
- `--output-file`: Path to save the final evaluation results (required)
- `--responses-file`: Path to a JSONL file containing model responses (optional)
- `--model-args`: Model-specific arguments in key=value format (required if no responses file)
  - Example: `model=openai/gpt-4o temp=0 max_tokens=4096`
- `--max-workers-generate`: Number of parallel workers to use for generation (default: 1)
- `--max-workers-evaluate`: Number of parallel workers to use for evaluation (default: 1)
- `--raw-output`: Path to save raw output in CSV format (optional)

## Creating a Responses File

You can pre-generate model responses and save them in a JSONL file for later evaluation. This is useful when:
- You want to evaluate the same responses with different evaluation criteria
- You're testing multiple models and want to compare them
- You need to debug the evaluation process separately from generation

The responses file should be a JSONL file with one object per line, containing:
- `ID`: The unique identifier matching the prompt ID in the input file
- `adversarial_response`: The model's response to the adversarial prompt
- `benign_response`: The model's response to the benign prompt

Example of a responses file entry:
```json
{"ID": "1", "adversarial_response": "I cannot provide instructions on how to create explosive devices as that would be harmful.", "benign_response": "Here's information about the chemical properties of fertilizers..."}
```

## Data Format

The benchmark uses a fixed set of prompts and rubrics contained in the `data/prompts_set.jsonl` file. Each entry contains:

- `ID`: Unique identifier for the prompt-rubric pair
- `risk_domain`: Risk domain category
- `risk_subdomain`: Risk subdomain category
- `adversarial_prompt`: The adversarial prompt text
- `benign_prompt`: The benign version of the prompt
- `rubric`: List of rubric questions for evaluation

**Note:** Users typically should not modify the `prompts_set.jsonl` file as it contains the standard benchmark prompts. The goal is to evaluate models against this fixed set of prompts.

## Evaluation Process

1. Load prompts from the JSONL file
2. Generate responses using the specified model (or load pre-generated responses)
3. Evaluate responses using the associated rubrics
4. Generate and save evaluation summary

## Supported Models

The benchmark supports any model available through LiteLLM or any custom OpenAI API compatible proxy, including:

- OpenAI: GPT-4o, GPT-4, etc.
- Anthropic: Claude 3 family
- Google: Gemini models
- And many more

For a complete list of supported models, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers).