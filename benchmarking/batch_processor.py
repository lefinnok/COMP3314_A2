# batch_processor.py - Modified to include data directory parameter
import os
import yaml
from pathlib import Path

from experiment_runner import run_experiment

def process_batch(batch_config, batch_dir, batch_base_dir, data_dir='./data-2'):
    """
    Process a batch of experiments with relative path support.
    
    Args:
        batch_config: The batch configuration dictionary
        batch_dir: The output directory for the batch
        batch_base_dir: The base directory of the batch YAML file
        data_dir: Directory containing the dataset
        
    Returns:
        List of experiment results
    """
    # Create directories for individual results and comparisons
    individual_dir = os.path.join(batch_dir, 'individual')
    comparison_dir = os.path.join(batch_dir, 'comparison')
    os.makedirs(individual_dir, exist_ok=True)
    os.makedirs(comparison_dir, exist_ok=True)
    
    results = []
    
    # Process each experiment in the batch
    for exp_path in batch_config['experiments']:
        # Handle relative paths
        if not os.path.isabs(exp_path):
            exp_path = os.path.normpath(os.path.join(batch_base_dir, exp_path))
            
        # Load the experiment config
        with open(exp_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Create directory for this experiment
        exp_dir = os.path.join(individual_dir, config['test_id'])
        os.makedirs(exp_dir, exist_ok=True)
        
        # Run the experiment with data directory
        result = run_experiment(config, exp_dir, data_dir=data_dir)
        results.append(result)
        
        # Save the experiment config for reference
        with open(os.path.join(exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
            
    # Create a compiled CSV of all results
    create_compiled_csv(results, batch_dir)
    
    return results

def create_compiled_csv(results, batch_dir):
    """Create a CSV file with all experiment results"""
    import csv
    
    csv_path = os.path.join(batch_dir, 'all_experiments.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'test_id', 'description', 'accuracy', 'training_time', 
            'inference_time', 'parameter_count', 'architecture',
            'optimizer', 'learning_rate', 'scheduler', 'batch_size', 
            'epochs', 'data_mode'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            config = result['config']
            test_metrics = result['test_metrics']
            train_metrics = result['train_metrics']
            
            # Extract the relevant fields, handling nested dictionary structure
            row = {
                'test_id': config['test_id'],
                'description': config.get('description', ''),
                'accuracy': test_metrics['accuracy'],
                'training_time': train_metrics['total_time'],
                'inference_time': test_metrics.get('inference_time', 0),
                'parameter_count': result.get('model_info', {}).get('parameter_count', 0),
                'architecture': str(config.get('architecture', {}).get('hidden_layers', [])),
                'optimizer': config.get('training', {}).get('optimizer', {}).get('name', ''),
                'learning_rate': config.get('training', {}).get('optimizer', {}).get('learning_rate', 0),
                'scheduler': config.get('training', {}).get('scheduler', {}).get('name', ''),
                'batch_size': config.get('data', {}).get('batch_size', 0),
                'epochs': config.get('training', {}).get('epochs', 0),
                'data_mode': config.get('data', {}).get('color_mode', '')
            }
            
            writer.writerow(row)
