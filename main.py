import argparse
from src.data_loader import DataLoader
from src.evaluator import Evaluator
from src.models.litellm import LiteLLMModel
from dotenv import load_dotenv
import json
import csv

def parse_provider_args(provider_args):
    """Parse key-value pairs from --provider-args."""
    args_dict = {}
    if provider_args:
        for arg in provider_args:
            key, value = arg.split('=')
            args_dict[key] = value
    return args_dict

def save_raw_output(data_loader, results, output_file, summary=None):
    """Save raw data, model responses, and judge scores to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'adversarial_prompt', 'benign_prompt', 'risk_domain', 'risk_subdomain', 
                     'adversarial_response', 'adversarial_overall_score', 
                     'benign_refusal', 'benign_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Get all prompt-rubric pairs
        pairs = data_loader.get_prompt_rubric_pairs()
        
        # Get all responses
        adversarial_responses = data_loader.get_responses("adversarial")
        benign_responses = data_loader.get_responses("benign")
        
        # Try to extract score from detailed results if summary is available
        pair_scores = {}
        benign_refusals = {}
        
        if summary and 'detailed_results' in summary:
            detailed_results = summary['detailed_results']
            for pair_id, pair_data in detailed_results.items():
                # Convert string ID to integer for consistent handling
                try:
                    int_pair_id = int(pair_id)
                    
                    # Extract adversarial scores
                    if 'adversarial' in pair_data:
                        adv_data = pair_data['adversarial']
                        if 'overall_score' in adv_data:
                            pair_scores[int_pair_id] = adv_data['overall_score']
                    
                    # Extract benign refusals
                    if 'benign' in pair_data and 'response_evaluations' in pair_data['benign'] and pair_data['benign']['response_evaluations']:
                        # Check the first response evaluation for refusal
                        refused = pair_data['benign']['response_evaluations'][0].get('refused', False)
                        benign_refusals[int_pair_id] = refused
                except (ValueError, TypeError):
                    print(f"Warning: Invalid ID format '{pair_id}' in detailed results, skipping")
        
        for pair in pairs:
            row = {
                'ID': pair.id,
                'adversarial_prompt': pair.adversarial_prompt,
                'benign_prompt': pair.benign_prompt,
                'risk_domain': pair.risk_domain,
                'risk_subdomain': pair.risk_subdomain
            }
            
            # Add adversarial response and refusal status
            if pair.id in adversarial_responses:
                row['adversarial_response'] = adversarial_responses[pair.id]
            else:
                row['adversarial_response'] = ""
            
            # Add benign response and refusal status
            benign_refusal = benign_refusals.get(pair.id, False)
            if pair.id in benign_responses:
                row['benign_response'] = benign_responses[pair.id]
            else:
                row['benign_response'] = ""
                
            row['benign_refusal'] = str(benign_refusal)
            
            # Add adversarial score
            score = None
            
            if pair.id in pair_scores:
                score = pair_scores[pair.id]
            elif pair.id in results and "adversarial" in results[pair.id]:
                try:
                    adversarial_result = results[pair.id]["adversarial"]
                    
                    # Try attribute access first
                    if hasattr(adversarial_result, 'overall_score'):
                        score = adversarial_result.overall_score
                    # If it fails, try all the fallback methods
                    elif hasattr(adversarial_result, '__dict__'):
                        score = adversarial_result.__dict__.get('overall_score')
                    
                    # If we still don't have a score, try as a last resort
                    if score is None and hasattr(adversarial_result, 'response_evaluations'):
                        responses = adversarial_result.response_evaluations
                        if responses:
                            scores = []
                            for resp in responses:
                                if hasattr(resp, 'final_score'):
                                    scores.append(resp.final_score)
                            if scores:
                                score = sum(scores) / len(scores)
                except Exception as e:
                    print(f"Error getting score for pair {pair.id}: {e}")
            
            # Ensure score is a string for CSV
            row['adversarial_overall_score'] = str(score) if score is not None else ""
            
            # Write each row to the CSV
            writer.writerow(row)
    
    print(f"Raw output saved to {output_file}")

def run_pipeline(args):
    """Run the full pipeline."""

    # Load data
    data_loader = DataLoader(args.input_file)
    data_loader.load_data()

    # Load or generate responses
    if args.responses_file:
        print(f"Loading responses from {args.responses_file}")
        data_loader.load_responses(args.responses_file)
    else:
        if not args.model_args:
            raise ValueError("You must specify a --model-args if generating responses.")
        
        provider_args = parse_provider_args(args.model_args)    
        model = LiteLLMModel(**provider_args)
        
        # Generate responses for each prompt
        data_loader.generate_adversarial_responses(model, args.max_workers_generate)
        data_loader.generate_benign_responses(model, args.max_workers_generate)

    # Evaluate responses
    evaluator = Evaluator(data_loader)
    results = evaluator.evaluate_all_pairs(args.max_workers_evaluate)
    
    # Generate summary
    summary = evaluator.get_evaluation_summary(results)

    
    # Save summary scores
    with open(args.output_file, 'w') as f:
        f.write(json.dumps(summary, indent=4))

    print(f"Evaluation complete. Results saved to {args.output_file}")

    # Save raw output if specified
    if args.raw_output:
        save_raw_output(data_loader, results, args.raw_output, summary)

def main():
    load_dotenv(dotenv_path="./.env",override=True)
    parser = argparse.ArgumentParser(description="LLM Adversarial Prompt Evaluation Tool")

    parser.add_argument('--input-file', type=str, required=True,
                       help="Path to the JSONL file containing prompts and rubrics.")
    parser.add_argument('--output-file', type=str, required=True,
                            help="Path to save the final evaluation results.")
    parser.add_argument('--responses-file', type=str,
                            help="Path to the JSON file containing model responses.")
    parser.add_argument('--model-args', type=str, nargs='*',
                        help="model-specific arguments in key=value format.")
    parser.add_argument('--max-workers-generate', type=int, default=1,
                            help="Number of parallel workers to use for generation. Only used if --responses-file is not provided.")
    parser.add_argument('--max-workers-evaluate', type=int, default=1,
                            help="Number of parallel workers to use.")
    parser.add_argument('--raw-output', type=str,
                            help="Path to save raw output.")

    args = parser.parse_args()

    run_pipeline(args)

if __name__ == '__main__':
    main() 