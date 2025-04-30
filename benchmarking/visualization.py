# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_comparisons(results, output_dir):
    """Generate comparison visualizations for a batch of experiments"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy comparison
    plot_batch_accuracy_comparison(results, output_dir)
    
    # Plot training curves comparison
    plot_batch_training_curves(results, output_dir)
    
    # Plot efficiency metrics
    plot_efficiency_metrics(results, output_dir)
    
    # Plot per-class performance comparison
    plot_class_performance_comparison(results, output_dir)
    
    # Plot hyperparameter effects
    plot_hyperparameter_effects(results, output_dir)

def plot_batch_accuracy_comparison(results, output_dir):
    """Plot overall accuracy comparison for all experiments"""
    # Extract relevant metrics
    test_ids = [r['config']['test_id'] for r in results]
    accuracies = [r['test_metrics']['accuracy'] for r in results]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracies)[::-1]  # Descending
    sorted_test_ids = [test_ids[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    # Plot accuracy comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(sorted_test_ids)), sorted_accuracies)
    plt.xticks(range(len(sorted_test_ids)), sorted_test_ids, rotation=45)
    plt.xlabel('Test ID')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy Comparison Across Experiments')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_accuracy_comparison.png'))
    plt.close()

def plot_batch_training_curves(results, output_dir):
    """Plot training curves comparison for all experiments"""
    plt.figure(figsize=(14, 10))
    
    # Plot training loss for each experiment
    plt.subplot(2, 2, 1)
    for result in results:
        test_id = result['config']['test_id']
        train_loss = result['train_metrics']['train_loss']
        epochs = range(1, len(train_loss) + 1)
        plt.plot(epochs, train_loss, marker='o', linestyle='-', label=test_id)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot training accuracy for each experiment
    plt.subplot(2, 2, 2)
    for result in results:
        test_id = result['config']['test_id']
        train_accuracy = result['train_metrics']['train_accuracy']
        epochs = range(1, len(train_accuracy) + 1)
        plt.plot(epochs, train_accuracy, marker='o', linestyle='-', label=test_id)
    plt.xlabel('Epochs')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Training Accuracy Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot learning rate progression for each experiment
    plt.subplot(2, 2, 3)
    for result in results:
        test_id = result['config']['test_id']
        learning_rates = result['train_metrics']['learning_rates']
        epochs = range(1, len(learning_rates) + 1)
        plt.plot(epochs, learning_rates, marker='o', linestyle='-', label=test_id)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Progression')
    plt.grid(True)
    plt.legend()
    
    # Plot epoch time for each experiment
    plt.subplot(2, 2, 4)
    for result in results:
        test_id = result['config']['test_id']
        epoch_times = result['train_metrics']['epoch_times']
        epochs = range(1, len(epoch_times) + 1)
        plt.plot(epochs, epoch_times, marker='o', linestyle='-', label=test_id)
    plt.xlabel('Epochs')
    plt.ylabel('Epoch Time (seconds)')
    plt.title('Epoch Time Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'batch_training_comparison.png'))
    plt.close()

def plot_efficiency_metrics(results, output_dir):
    """Plot efficiency metrics (accuracy vs. training time/parameter count)"""
    # Extract metrics
    test_ids = [r['config']['test_id'] for r in results]
    accuracies = [r['test_metrics']['accuracy'] for r in results]
    training_times = [r['train_metrics']['total_time'] for r in results]
    parameter_counts = [r['model_info']['parameter_count'] for r in results]
    
    # Create figure
    plt.figure(figsize=(14, 12))
    
    # Accuracy vs. Training Time
    plt.subplot(2, 1, 1)
    plt.scatter(training_times, accuracies, s=100, alpha=0.7)
    
    # Add test_id labels to points
    for i, test_id in enumerate(test_ids):
        plt.annotate(test_id, (training_times[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Training Time')
    plt.grid(True)
    
    # Accuracy vs. Parameter Count
    plt.subplot(2, 1, 2)
    # Convert to millions of parameters for readability
    param_millions = [p/1000000 for p in parameter_counts]
    plt.scatter(param_millions, accuracies, s=100, alpha=0.7)
    
    # Add test_id labels to points
    for i, test_id in enumerate(test_ids):
        plt.annotate(test_id, (param_millions[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Parameter Count (millions)')
    plt.ylabel('Test Accuracy (%)')
    plt.title('Accuracy vs. Model Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_comparison.png'))
    plt.close()

def plot_class_performance_comparison(results, output_dir):
    """Plot per-class performance comparison for all experiments"""
    # Prepare data
    test_ids = [r['config']['test_id'] for r in results]
    class_accuracies = [r['test_metrics']['class_accuracy'] for r in results]
    
    # Create grouped bar chart
    plt.figure(figsize=(15, 8))
    
    x = np.arange(10)  # 10 digit classes
    width = 0.8 / len(test_ids)  # Width of the bars
    
    # Plot bars for each test
    for i, (test_id, class_acc) in enumerate(zip(test_ids, class_accuracies)):
        offset = (i - len(test_ids)/2 + 0.5) * width
        plt.bar(x + offset, class_acc, width, label=test_id)
    
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy Comparison')
    plt.xticks(x, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    plt.legend()
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_accuracy_comparison.png'))
    plt.close()

def plot_hyperparameter_effects(results, output_dir):
    """Plot effects of various hyperparameters on model performance"""
    # Define hyperparameters to analyze
    hyperparams = [
        ('training.optimizer.learning_rate', 'Learning Rate'),
        ('training.epochs', 'Number of Epochs'),
        ('data.batch_size', 'Batch Size')
    ]
    
    for param_path, param_name in hyperparams:
        # Extract the specific hyperparameter and accuracy
        param_values = []
        accuracies = []
        test_ids = []
        
        for result in results:
            # Extract the parameter value using the path
            parts = param_path.split('.')
            value = result['config']
            try:
                for part in parts:
                    value = value.get(part)
                
                if value is not None:
                    param_values.append(value)
                    accuracies.append(result['test_metrics']['accuracy'])
                    test_ids.append(result['config']['test_id'])
            except (AttributeError, TypeError):
                # Skip if the parameter doesn't exist in this config
                continue
        
        if not param_values:
            continue  # Skip if no values found
        
        # Sort by parameter value
        sorted_indices = np.argsort(param_values)
        sorted_param_values = [param_values[i] for i in sorted_indices]
        sorted_accuracies = [accuracies[i] for i in sorted_indices]
        sorted_test_ids = [test_ids[i] for i in sorted_indices]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_param_values, sorted_accuracies, 'o-', linewidth=2, markersize=8)
        
        # Add test_id labels
        for i, test_id in enumerate(sorted_test_ids):
            plt.annotate(test_id, 
                        (sorted_param_values[i], sorted_accuracies[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel(param_name)
        plt.ylabel('Test Accuracy (%)')
        plt.title(f'Effect of {param_name} on Model Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'param_effect_{param_path.replace(".", "_")}.png'))
        plt.close()
