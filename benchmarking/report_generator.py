# report_generator.py
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_individual_report(result, output_dir):
    """Generate a detailed report for an individual experiment"""
    # Create report directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract relevant data
    config = result['config']
    train_metrics = result['train_metrics']
    test_metrics = result['test_metrics']
    
    # Generate markdown report
    report_path = os.path.join(output_dir, f"{config['test_id']}_report.md")
    
    with open(report_path, 'w') as f:
        # Header
        f.write(f"# Experiment Report: {config['test_id']}\n\n")
        f.write(f"## Description\n{config.get('description', 'No description provided')}\n\n")
        
        # Configuration
        f.write("## Configuration\n")
        f.write("```yaml\n")
        yaml_config = {
            'architecture': config.get('architecture', {}),
            'data': config.get('data', {}),
            'training': config.get('training', {})
        }
        f.write(yaml.dump(yaml_config))
        f.write("```\n\n")
        
        # Performance Summary
        f.write("## Performance Summary\n\n")
        f.write(f"- **Overall Accuracy**: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"- **Training Time**: {train_metrics['total_time']:.2f} seconds\n")
        f.write(f"- **Inference Time**: {test_metrics['inference_time']:.2f} seconds\n")
        f.write(f"- **Parameter Count**: {result['model_info']['parameter_count']:,}\n\n")
        
        # Training Progression
        f.write("## Training Progression\n\n")
        f.write("![Training Curves](training_curves.png)\n\n")
        
        # Test Performance
        f.write("## Test Performance\n\n")
        f.write("### Confusion Matrix\n\n")
        f.write("![Confusion Matrix](confusion_matrix.png)\n\n")
        
        f.write("### Per-Class Accuracy\n\n")
        f.write("![Class Accuracy](class_accuracy.png)\n\n")
        
        f.write("| Class | Accuracy (%) |\n")
        f.write("|-------|-------------|\n")
        for i, acc in enumerate(test_metrics['class_accuracy']):
            f.write(f"| {i} | {acc:.2f} |\n")
    
    # Generate visualizations for this report
    generate_individual_visualizations(result, output_dir)
    
    return report_path

def generate_individual_visualizations(result, output_dir):
    """Generate visualizations for an individual experiment report"""
    # Training curves
    plt.figure(figsize=(12, 8))
    
    # Training loss
    plt.subplot(2, 2, 1)
    epochs = range(1, len(result['train_metrics']['train_loss']) + 1)
    plt.plot(epochs, result['train_metrics']['train_loss'], 'b-o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    
    # Training accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, result['train_metrics']['train_accuracy'], 'g-o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy')
    plt.grid(True)
    
    # Learning rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, result['train_metrics']['learning_rates'], 'r-o')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    
    # Epoch time
    plt.subplot(2, 2, 4)
    plt.plot(epochs, result['train_metrics']['epoch_times'], 'm-o')
    plt.xlabel('Epochs')
    plt.ylabel('Time (seconds)')
    plt.title('Epoch Time')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    confusion_matrix = result['test_metrics']['confusion_matrix']
    
    im = plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar(im)
    
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.yticks(tick_marks, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Per-class accuracy
    plt.figure(figsize=(12, 6))
    class_accuracy = result['test_metrics']['class_accuracy']
    
    bars = plt.bar(range(10), class_accuracy)
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(10), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.grid(True, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy.png'))
    plt.close()

def generate_batch_report(results, output_dir):
    """Generate a comprehensive report comparing all experiments in a batch"""
    # Create report directory
    comparison_dir = os.path.join(output_dir, 'comparison')
    os.makedirs(comparison_dir, exist_ok=True)
    
    # Sort results by accuracy
    sorted_results = sorted(results, 
                           key=lambda x: x['test_metrics']['accuracy'],
                           reverse=True)
    
    # Generate markdown report
    report_path = os.path.join(output_dir, "batch_comparison_report.md")
    
    with open(report_path, 'w') as f:
        # Header
        f.write("# Batch Experiment Comparison Report\n\n")
        
        # Overview of experiments
        f.write("## Experiments Overview\n\n")
        f.write("| Test ID | Description | Accuracy (%) | Training Time (s) | Parameters |\n")
        f.write("|---------|-------------|--------------|-------------------|------------|\n")
        
        for result in sorted_results:
            test_id = result['config']['test_id']
            desc = result['config'].get('description', 'No description')
            acc = result['test_metrics']['accuracy']
            time = result['train_metrics']['total_time']
            params = result['model_info']['parameter_count']
            
            f.write(f"| {test_id} | {desc} | {acc:.2f} | {time:.2f} | {params:,} |\n")
        
        f.write("\n")
        
        # Best performing model
        best_result = sorted_results[0]
        f.write("## Best Performing Model\n\n")
        f.write(f"- **Test ID**: {best_result['config']['test_id']}\n")
        f.write(f"- **Description**: {best_result['config'].get('description', 'No description')}\n")
        f.write(f"- **Accuracy**: {best_result['test_metrics']['accuracy']:.2f}%\n")
        f.write(f"- **Training Time**: {best_result['train_metrics']['total_time']:.2f} seconds\n\n")
        
        f.write("### Configuration\n")
        f.write("```yaml\n")
        yaml_config = {
            'architecture': best_result['config'].get('architecture', {}),
            'data': best_result['config'].get('data', {}),
            'training': best_result['config'].get('training', {})
        }
        f.write(yaml.dump(yaml_config))
        f.write("```\n\n")
        
        # Comparative Visualizations
        f.write("## Comparative Analysis\n\n")
        
        f.write("### Accuracy Comparison\n")
        f.write("![Accuracy Comparison](comparison/batch_accuracy_comparison.png)\n\n")
        
        f.write("### Training Convergence\n")
        f.write("![Training Comparison](comparison/batch_training_comparison.png)\n\n")
        
        f.write("### Efficiency Metrics\n")
        f.write("![Efficiency Metrics](comparison/efficiency_comparison.png)\n\n")
        
        f.write("### Per-Class Performance\n")
        f.write("![Class Performance](comparison/class_accuracy_comparison.png)\n\n")
        
        # Hyperparameter Analysis
        f.write("## Hyperparameter Analysis\n\n")
        
        # Learning rate effect
        f.write("### Effect of Learning Rate\n")
        f.write("![Learning Rate Effect](comparison/param_effect_training_optimizer_learning_rate.png)\n\n")
        
        # Epochs effect
        f.write("### Effect of Training Epochs\n")
        f.write("![Epochs Effect](comparison/param_effect_training_epochs.png)\n\n")
        
        # Batch size effect
        f.write("### Effect of Batch Size\n")
        f.write("![Batch Size Effect](comparison/param_effect_data_batch_size.png)\n\n")
        
        # Key Findings and Insights
        f.write("## Key Findings\n\n")
        f.write("- The highest accuracy achieved was ")
        f.write(f"{best_result['test_metrics']['accuracy']:.2f}% ")
        f.write(f"by {best_result['config']['test_id']}.\n")
        
        # Check if there are significant differences in per-class performance
        class_std = np.std([r['test_metrics']['class_accuracy'] for r in results], axis=0)
        problematic_classes = np.where(class_std > 5.0)[0]  # Classes with high variance across models
        
        if len(problematic_classes) > 0:
            f.write("- There were significant differences in model performance for classes: ")
            f.write(", ".join([str(c) for c in problematic_classes]))
            f.write(". These classes may require special attention.\n")
        
        # Compare training times
        fastest_result = min(results, key=lambda x: x['train_metrics']['total_time'])
        slowest_result = max(results, key=lambda x: x['train_metrics']['total_time'])
        
        f.write(f"- Training times varied from ")
        f.write(f"{fastest_result['train_metrics']['total_time']:.2f}s ")
        f.write(f"({fastest_result['config']['test_id']}) to ")
        f.write(f"{slowest_result['train_metrics']['total_time']:.2f}s ")
        f.write(f"({slowest_result['config']['test_id']}).\n")
        
        # Recommendations
        f.write("\n## Recommendations\n\n")
        
        # Best configuration
        f.write("- **Best Configuration**: Based on the experiments, the configuration from ")
        f.write(f"{best_result['config']['test_id']} provided the best accuracy.\n")
        
        # Learning rate recommendations
        lr_values = [(r['config']['training']['optimizer']['learning_rate'], 
                     r['test_metrics']['accuracy']) 
                    for r in results 
                    if 'training' in r['config'] and 
                    'optimizer' in r['config']['training'] and 
                    'learning_rate' in r['config']['training']['optimizer']]
        
        if lr_values:
            lr_values.sort(key=lambda x: x[1], reverse=True)  # Sort by accuracy
            f.write(f"- **Learning Rate**: A learning rate of {lr_values[0][0]} appeared to be most effective.\n")
        
        # Other recommendations based on data analysis
        f.write("- **Further Exploration**: Consider additional experiments with modified network ")
        f.write("architecture or data augmentation techniques to potentially improve accuracy further.\n")
    
    return report_path
