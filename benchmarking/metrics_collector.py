# metrics_collector.py
import os
import csv
import json
import numpy as np

class MetricsCollector:
    """Class to collect and save metrics during experiments"""
    
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize CSV files
        self._init_training_metrics_csv()
        self._init_test_metrics_csv()
    
    def _init_training_metrics_csv(self):
        """Initialize the training metrics CSV file"""
        csv_path = os.path.join(self.output_dir, 'training_metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'train_accuracy', 
                         'val_loss', 'val_accuracy', 'learning_rate', 'epoch_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def _init_test_metrics_csv(self):
        """Initialize the test metrics CSV file"""
        csv_path = os.path.join(self.output_dir, 'test_metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['metric', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
    
    def update_training_metrics(self, epoch, train_loss, train_accuracy, 
                               val_loss, val_accuracy, learning_rate, epoch_time):
        """Update training metrics for the current epoch"""
        csv_path = os.path.join(self.output_dir, 'training_metrics.csv')
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'train_loss', 'train_accuracy', 
                                                       'val_loss', 'val_accuracy', 
                                                       'learning_rate', 'epoch_time'])
            writer.writerow({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'learning_rate': learning_rate,
                'epoch_time': epoch_time
            })
    
    def update_test_metrics(self, accuracy, class_accuracy, confusion_matrix, inference_time):
        """Update test metrics"""
        csv_path = os.path.join(self.output_dir, 'test_metrics.csv')
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['metric', 'value'])
            writer.writerow({'metric': 'accuracy', 'value': accuracy})
            writer.writerow({'metric': 'inference_time', 'value': inference_time})
        
        # Save confusion matrix
        np.save(os.path.join(self.output_dir, 'confusion_matrix.npy'), confusion_matrix)
        
        # Save class accuracy
        with open(os.path.join(self.output_dir, 'class_accuracy.json'), 'w') as f:
            json.dump(class_accuracy, f)
    
    def save_class_metrics(self, class_metrics):
        """Save per-class metrics"""
        csv_path = os.path.join(self.output_dir, 'class_metrics.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['class', 'accuracy', 'sample_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for metrics in class_metrics:
                writer.writerow(metrics)
