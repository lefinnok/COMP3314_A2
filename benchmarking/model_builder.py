# model_builder.py - Modified to support per-layer activations
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_channels, hidden_layers, dropout_rates=None, activations=None):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_layers = hidden_layers
        
        # Default dropout rates if not provided
        if dropout_rates is None:
            dropout_rates = [0.5, 0.3] + [0.0] * (len(hidden_layers) - 2)
        
        # Ensure we have enough dropout rates
        if len(dropout_rates) < len(hidden_layers):
            dropout_rates.extend([0.0] * (len(hidden_layers) - len(dropout_rates)))
        
        # Default activations if not provided (use ReLU for all layers)
        if activations is None:
            activations = ['relu'] * len(hidden_layers)
        elif isinstance(activations, str):
            # If a single string is provided, apply the same activation to all layers
            activations = [activations] * len(hidden_layers)
        
        # Ensure we have enough activations
        if len(activations) < len(hidden_layers):
            activations.extend(['relu'] * (len(hidden_layers) - len(activations)))
        
        # Build network
        layers = []
        
        # Flatten layer
        layers.append(nn.Flatten())
        
        # Input layer
        in_features = 32 * 32 * input_channels
        
        # Hidden layers
        for i, (out_features, dropout_rate, activation) in enumerate(zip(hidden_layers, dropout_rates, activations)):
            # Linear layer
            layers.append(nn.Linear(in_features, out_features))
            
            # Activation layer
            layers.append(self._get_activation(activation))
            
            # Dropout layer
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            in_features = out_features
        
        # Output layer - no activation or dropout needed for the final layer
        layers.append(nn.Linear(in_features, 10))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation):
        """Get the appropriate activation function based on the name"""
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation.lower() == 'selu':
            return nn.SELU()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'elu':
            return nn.ELU()
        elif activation.lower() == 'gelu':
            return nn.GELU()
        else:
            # Default to ReLU
            return nn.ReLU()
    
    def forward(self, x):
        return self.network(x)

def build_model(arch_config, input_channels=1):
    """Build a model based on architecture configuration"""
    hidden_layers = arch_config.get('hidden_layers', [512, 256])
    dropout_rates = arch_config.get('dropout_rates', [0.5, 0.3])
    activations = arch_config.get('activations', 'relu')
    
    return MLP(input_channels, hidden_layers, dropout_rates, activations)
