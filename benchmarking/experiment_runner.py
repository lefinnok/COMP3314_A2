# experiment_runner.py
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import transforms, datasets
import csv

from model_builder import build_model
from metrics_collector import MetricsCollector

class ExperimentRunner:
    def __init__(self, config, output_dir, data_dir='./data-2'):
        self.config = config
        self.output_dir = output_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Prepare directories
        os.makedirs(output_dir, exist_ok=True)
        
        # Create metrics collector
        self.metrics_collector = MetricsCollector(output_dir)
        
    def prepare_data(self):
        """Prepare data loaders based on configuration"""
        data_config = self.config.get('data', {})
        
        # Set up transformations
        transform_train = self._get_transforms(data_config, is_training=True)
        transform_test = self._get_transforms(data_config, is_training=False)
        
        # Load datasets using the provided data directory
        train_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'train'),
            transform=transform_train
        )
        
        test_dataset = datasets.ImageFolder(
            os.path.join(self.data_dir, 'test'),
            transform=transform_test
        )
        
        # Create data loaders
        batch_size = data_config.get('batch_size', 128)
        num_workers = 4  # Should be configurable
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True, 
            num_workers=num_workers
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False, 
            num_workers=num_workers
        )
        
        return train_loader, test_loader, len(train_dataset)
    
    def _get_transforms(self, data_config, is_training):
        """Create data transformations based on configuration"""
        color_mode = data_config.get('color_mode', 'grayscale')
        num_channels = 1 if color_mode == 'grayscale' else 3
        
        # Basic transforms
        transform_list = []
        
        # Grayscale conversion if needed
        if color_mode == 'grayscale':
            transform_list.append(transforms.Grayscale(num_output_channels=1))
        
        # Add augmentation for training
        if is_training:
            aug_config = data_config.get('augmentation', {})
            
            # Random crop and resize
            if aug_config.get('random_crop', {}).get('enabled', False):
                size = aug_config['random_crop'].get('size', [32, 32])
                padding = aug_config['random_crop'].get('padding', 4)
                transform_list.append(transforms.RandomCrop(size=size, padding=padding))
            
            # Random affine transform
            if aug_config.get('random_affine', {}).get('enabled', False):
                degrees = aug_config['random_affine'].get('degrees', 10)
                translate = tuple(aug_config['random_affine'].get('translate', [0, 0.1]))
                transform_list.append(transforms.RandomAffine(degrees=degrees, translate=translate))
            
            # Random rotation
            if aug_config.get('random_rotation', {}).get('enabled', False):
                degrees = aug_config['random_rotation'].get('degrees', 10)
                transform_list.append(transforms.RandomRotation(degrees=degrees))
            
            # Random horizontal flip
            if aug_config.get('random_horizontal_flip', {}).get('enabled', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            
            # Random perspective
            if aug_config.get('random_perspective', {}).get('enabled', False):
                distortion_scale = aug_config['random_perspective'].get('distortion_scale', 0.5)
                p = aug_config['random_perspective'].get('probability', 0.5)
                transform_list.append(transforms.RandomPerspective(distortion_scale=distortion_scale, p=p))
            
            # Color jitter
            if aug_config.get('color_jitter', {}).get('enabled', False) and color_mode == 'rgb':
                brightness = aug_config['color_jitter'].get('brightness', 0.2)
                contrast = aug_config['color_jitter'].get('contrast', 0.2)
                saturation = aug_config['color_jitter'].get('saturation', 0.2)
                hue = aug_config['color_jitter'].get('hue', 0.1)
                transform_list.append(transforms.ColorJitter(
                    brightness=brightness, contrast=contrast, 
                    saturation=saturation, hue=hue))
            
            # Gaussian blur
            if aug_config.get('gaussian_blur', {}).get('enabled', False):
                kernel_size = aug_config['gaussian_blur'].get('kernel_size', 5)
                sigma = aug_config['gaussian_blur'].get('sigma', [0.1, 2.0])
                transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))
        
        # Common transforms for both training and testing
        transform_list.extend([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        
        # Normalization
        norm_values = data_config.get('normalization', [0.5, 0.5])
        transform_list.append(transforms.Normalize([norm_values[0]] * num_channels, 
                                                  [norm_values[1]] * num_channels))
        
        # Random erasing (applied after tensor conversion)
        if is_training and aug_config.get('random_erasing', {}).get('enabled', False):
            p = aug_config['random_erasing'].get('probability', 0.5)
            scale = tuple(aug_config['random_erasing'].get('scale', [0.02, 0.33]))
            ratio = tuple(aug_config['random_erasing'].get('ratio', [0.3, 3.3]))
            transform_list.append(transforms.RandomErasing(p=p, scale=scale, ratio=ratio))
        
        return transforms.Compose(transform_list)
    
    def build_model(self):
        """Build the model based on configuration"""
        arch_config = self.config.get('architecture', {})
        color_mode = self.config.get('data', {}).get('color_mode', 'grayscale')
        num_channels = 1 if color_mode == 'grayscale' else 3
        
        model = build_model(arch_config, num_channels)
        model = model.to(self.device)
        
        return model
    
    def configure_training(self, model):
        """Configure training components based on configuration"""
        train_config = self.config.get('training', {})
        
        # Configure loss function
        criterion = nn.CrossEntropyLoss()
        
        # Configure optimizer
        optimizer_config = train_config.get('optimizer', {})
        optimizer_name = optimizer_config.get('name', 'adam')
        learning_rate = optimizer_config.get('learning_rate', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0)
        
        if optimizer_name.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                                 momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'rmsprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Configure learning rate scheduler
        scheduler_config = train_config.get('scheduler', {})
        scheduler_name = scheduler_config.get('name', 'step_lr')
        
        if scheduler_name.lower() == 'step_lr':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.7)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == 'cosine':
            t_max = scheduler_config.get('t_max', train_config.get('epochs', 15))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        elif scheduler_name.lower() == 'plateau':
            patience = scheduler_config.get('patience', 3)
            factor = scheduler_config.get('factor', 0.1)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
        
        return criterion, optimizer, scheduler
    
    def train(self, model, train_loader, criterion, optimizer, scheduler, train_size):
        """Train the model and collect metrics"""
        train_config = self.config.get('training', {})
        num_epochs = train_config.get('epochs', 15)
        
        # Initialize metrics
        train_loss = []
        train_accuracy = []
        val_loss = []
        val_accuracy = []
        learning_rates = []
        epoch_times = []
        
        # Training loop
        total_start_time = time.time()
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_corrects = 0
            epoch_start_time = time.time()
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Print progress
                if (i + 1) % 100 == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            # Calculate epoch statistics
            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            epoch_loss = running_loss / train_size
            epoch_acc = running_corrects / train_size * 100
            
            # Update learning rate scheduler
            current_lr = optimizer.param_groups[0]['lr']
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_loss)  # Pass validation loss for plateau scheduler
            else:
                scheduler.step()
            
            # Collect metrics
            train_loss.append(epoch_loss)
            train_accuracy.append(epoch_acc)
            learning_rates.append(current_lr)
            epoch_times.append(epoch_time)
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, ' 
                  f'Time: {epoch_time:.2f}s, LR: {current_lr:.6f}')
            
            # Save metrics for this epoch
            self.metrics_collector.update_training_metrics(
                epoch + 1, epoch_loss, epoch_acc, None, None, current_lr, epoch_time
            )
        
        total_end_time = time.time()
        total_train_time = total_end_time - total_start_time
        print(f'Training completed in {total_train_time:.2f} seconds')
        
        # Return collected metrics
        return {
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rates': learning_rates,
            'epoch_times': epoch_times,
            'total_time': total_train_time
        }
    
    def evaluate(self, model, test_loader):
        """Evaluate the model on test data and collect metrics"""
        model.eval()
        
        running_corrects = 0
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        all_preds = []
        all_labels = []
        
        # Time inference
        start_time = time.time()
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
                # Collect predictions and labels for confusion matrix
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Statistics
                running_corrects += torch.sum(preds == labels.data).item()
                
                # Per-class accuracy
                c = (preds == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        end_time = time.time()
        inference_time = end_time - start_time
        
        # Calculate metrics
        accuracy = running_corrects / sum(class_total) * 100
        
        # Calculate per-class accuracy
        class_accuracy = []
        for i in range(10):
            if class_total[i] > 0:
                class_accuracy.append(class_correct[i] / class_total[i] * 100)
            else:
                class_accuracy.append(0)
        
        # Create confusion matrix
        confusion_matrix = np.zeros((10, 10), dtype=int)
        for pred, label in zip(all_preds, all_labels):
            confusion_matrix[label, pred] += 1
        
        # Save test metrics
        self.metrics_collector.update_test_metrics(accuracy, class_accuracy, confusion_matrix, inference_time)
        
        # Save per-class metrics
        class_metrics = []
        for i in range(10):
            metrics = {
                'class': i,
                'accuracy': class_accuracy[i],
                'sample_count': class_total[i]
            }
            class_metrics.append(metrics)
        
        self.metrics_collector.save_class_metrics(class_metrics)
        
        return {
            'accuracy': accuracy,
            'class_accuracy': class_accuracy,
            'confusion_matrix': confusion_matrix,
            'inference_time': inference_time
        }
    
    def run(self):
        """Run the complete experiment"""
        print(f"Starting experiment: {self.config['test_id']}")
        
        # Prepare data
        train_loader, test_loader, train_size = self.prepare_data()
        
        # Build model
        model = self.build_model()
        print(model)
        
        # Configure training
        criterion, optimizer, scheduler = self.configure_training(model)
        
        # Train model
        train_metrics = self.train(model, train_loader, criterion, optimizer, scheduler, train_size)
        
        # Evaluate model
        test_metrics = self.evaluate(model, test_loader)
        
        # Save model
        torch.save(model.state_dict(), os.path.join(self.output_dir, 'model.pt'))
        
        # Get model info
        model_info = {
            'parameter_count': sum(p.numel() for p in model.parameters())
        }
        
        # Return results
        return {
            'config': self.config,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model_info': model_info,
            'output_dir': self.output_dir
        }

def run_experiment(config, output_dir, data_dir='./data-2'):
    """Run an experiment with the given configuration"""
    runner = ExperimentRunner(config, output_dir, data_dir)
    return runner.run()
