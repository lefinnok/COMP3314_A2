# benchmark.py - Modified to include data directory argument
import os
import sys
import argparse
import yaml
import glob
from pathlib import Path

from experiment_runner import run_experiment
from batch_processor import process_batch
from report_generator import generate_individual_report, generate_batch_report
from visualization import generate_comparisons

def parse_args():
    parser = argparse.ArgumentParser(description='MLP Benchmarking Framework')
    
    # Main operation modes
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config', type=str, help='Path to single experiment YAML config')
    group.add_argument('--batch', type=str, help='Path to batch YAML config')
    group.add_argument('--config-dir', type=str, help='Directory containing YAML configs')
    group.add_argument('--report-only', action='store_true', help='Generate reports from existing results')
    
    # Additional options
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--data-dir', type=str, default='./data-2', help='Directory containing the dataset')
    parser.add_argument('--recursive', action='store_true', help='Recursively process directories')
    parser.add_argument('--results-dir', type=str, help='Directory containing results for report generation')
    
    return parser.parse_args()

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    args = parse_args()
    
    if args.config:
        # Run a single experiment
        config = load_yaml(args.config)
        result_dir = os.path.join(args.output_dir, config['test_id'])
        os.makedirs(result_dir, exist_ok=True)
        
        result = run_experiment(config, result_dir, data_dir=args.data_dir)
        generate_individual_report(result, result_dir)
        
    elif args.batch:
        # Process a batch of experiments
        batch_config = load_yaml(args.batch)
        batch_dir = os.path.join(args.output_dir, batch_config['batch_id'])
        os.makedirs(batch_dir, exist_ok=True)
        
        # Process the batch with relative path support
        batch_base_dir = os.path.dirname(os.path.abspath(args.batch))
        results = process_batch(batch_config, batch_dir, batch_base_dir, data_dir=args.data_dir)
        
        # Generate batch reports and visualizations
        generate_batch_report(results, batch_dir)
        generate_comparisons(results, os.path.join(batch_dir, 'comparison'))
        
    elif args.config_dir:
        # Process all configs in a directory
        dir_path = args.config_dir
        pattern = '**/*.yaml' if args.recursive else '*.yaml'
        config_files = glob.glob(os.path.join(dir_path, pattern), recursive=args.recursive)
        
        # Create a batch from all found config files
        batch_id = os.path.basename(os.path.normpath(dir_path))
        results = []
        
        for config_file in config_files:
            config = load_yaml(config_file)
            result_dir = os.path.join(args.output_dir, batch_id, 'individual', config['test_id'])
            os.makedirs(result_dir, exist_ok=True)
            
            result = run_experiment(config, result_dir, data_dir=args.data_dir)
            generate_individual_report(result, result_dir)
            results.append(result)
        
        # Generate batch reports
        batch_dir = os.path.join(args.output_dir, batch_id)
        generate_batch_report(results, batch_dir)
        generate_comparisons(results, os.path.join(batch_dir, 'comparison'))
        
    elif args.report_only:
        # Generate reports from existing results
        if not args.results_dir:
            print("Error: --results-dir is required with --report-only")
            sys.exit(1)
            
        # Implement report generation from existing results
        # This would need to load the saved metrics and generate reports
        pass

if __name__ == "__main__":
    main()
