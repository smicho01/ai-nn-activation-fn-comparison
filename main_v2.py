import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

# Configuration and utilities
class ActivationType(Enum):
    NONE = "none"
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"

@dataclass
class TrainingConfig:
    epochs: int = 1000
    learning_rate: float = 0.001
    batch_size: int = 64
    print_interval: int = 100
    seed: int = 42

@dataclass
class NetworkConfig:
    input_dim: int = 1
    hidden_dim: int = 64
    output_dim: int = 1
    num_layers: int = 5
    activation: ActivationType = ActivationType.RELU

class DeviceManager:
    @staticmethod
    def get_optimal_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

class DataGenerator:
    def __init__(self, seed: int = 42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = DeviceManager.get_optimal_device()
        print(f"🔧 Using device: {self.device}")
    
    def generate_polynomial_data(self, 
                                power: int = 3, 
                                x_range: Tuple[float, float] = (-10, 10),
                                num_points: int = 1000,
                                noise_std: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate polynomial data y = x^power with optional noise"""
        x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1)
        y = x ** power
        
        if noise_std > 0:
            noise = torch.randn_like(y) * noise_std
            y += noise
        
        return x.to(self.device), y.to(self.device)

class ActivationFactory:
    @staticmethod
    def create_activation(activation_type: ActivationType) -> Optional[nn.Module]:
        activation_map = {
            ActivationType.NONE: None,
            ActivationType.RELU: nn.ReLU(),
            ActivationType.TANH: nn.Tanh(),
            ActivationType.SIGMOID: nn.Sigmoid(),
            ActivationType.LEAKY_RELU: nn.LeakyReLU(0.1)
        }
        return activation_map.get(activation_type)

class DynamicNeuralNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super(DynamicNeuralNetwork, self).__init__()
        self.config = config
        self.layers = self._build_network()
        self.model = nn.Sequential(*self.layers)
        
    def _build_network(self) -> list:
        layers = []
        activation_fn = ActivationFactory.create_activation(self.config.activation)
        
        # Input layer
        layers.append(nn.Linear(self.config.input_dim, self.config.hidden_dim))
        if activation_fn:
            layers.append(activation_fn)
        
        # Hidden layers
        for _ in range(self.config.num_layers - 2):
            layers.append(nn.Linear(self.config.hidden_dim, self.config.hidden_dim))
            if activation_fn:
                layers.append(activation_fn)
        
        # Output layer
        layers.append(nn.Linear(self.config.hidden_dim, self.config.output_dim))
        
        return layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def get_architecture_summary(self) -> str:
        total_params = sum(p.numel() for p in self.parameters())
        return f"🏗️  Architecture: {self.config.activation.value} | Layers: {self.config.num_layers} | Params: {total_params:,}"

class NeuralTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = DeviceManager.get_optimal_device()
        self.loss_history = []
    
    def train_model(self, 
                   model: nn.Module, 
                   x_train: torch.Tensor, 
                   y_train: torch.Tensor,
                   criterion: nn.Module = None,
                   optimizer_class: type = optim.Adam) -> nn.Module:
        
        model = model.to(self.device)
        criterion = criterion or nn.MSELoss()
        optimizer = optimizer_class(model.parameters(), lr=self.config.learning_rate)
        
        print(f"🚀 Training started for {self.config.epochs} epochs...")
        
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            if (epoch + 1) % self.config.print_interval == 0:
                print(f"📊 Epoch [{epoch+1:4d}/{self.config.epochs}] | Loss: {loss.item():.6f}")
        
        print("✅ Training completed!")
        return model
    
    def get_final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float('inf')

class ResultVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.device = DeviceManager.get_optimal_device()
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def plot_prediction_comparison(self, 
                                 model: nn.Module, 
                                 title: str,
                                 x_range: Tuple[float, float] = (-10, 10),
                                 true_function: Callable = lambda x: x**3,
                                 num_points: int = 1000) -> None:
        
        model.eval()
        with torch.no_grad():
            x_plot = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1).to(self.device)
            y_true = true_function(x_plot)
            y_pred = model(x_plot)
        
        # Convert to numpy for plotting
        x_np = x_plot.cpu().numpy().flatten()
        y_true_np = y_true.cpu().numpy().flatten()
        y_pred_np = y_pred.cpu().numpy().flatten()
        
        # Calculate metrics
        mse = np.mean((y_true_np - y_pred_np) ** 2)
        
        plt.figure(figsize=self.figsize)
        plt.plot(x_np, y_true_np, label='True Function $x^3$', color='navy', linewidth=2, alpha=0.8)
        plt.plot(x_np, y_pred_np, label='Neural Network Prediction', color='crimson', linewidth=2, linestyle='--')
        
        plt.title(f'{title}\nMSE: {mse:.6f}', fontsize=14, fontweight='bold')
        plt.xlabel('Input (x)', fontsize=12)
        plt.ylabel('Output (y)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_training_curves(self, trainers: list, labels: list) -> None:
        plt.figure(figsize=self.figsize)
        
        for trainer, label in zip(trainers, labels):
            epochs = range(1, len(trainer.loss_history) + 1)
            plt.plot(epochs, trainer.loss_history, label=f'{label} (Final: {trainer.get_final_loss():.6f})', linewidth=2)
        
        plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.show()

class NeuralNetworkExperiment:
    def __init__(self, training_config: TrainingConfig = None):
        self.training_config = training_config or TrainingConfig()
        self.data_generator = DataGenerator(self.training_config.seed)
        self.visualizer = ResultVisualizer()
        self.results = {}
    
    def run_experiment(self):
        print("🧪 Starting Neural Network Comparison Experiment")
        print("=" * 60)
        
        # Generate training data
        x_train, y_train = self.data_generator.generate_polynomial_data(power=3)
        print(f"📊 Generated {len(x_train)} training samples")
        
        # Define network configurations
        configs = [
            ("Without Activation", NetworkConfig(activation=ActivationType.NONE)),
            ("With ReLU Activation", NetworkConfig(activation=ActivationType.RELU)),
            ("With Tanh Activation", NetworkConfig(activation=ActivationType.TANH)),
            ("With Sigmoid Activation", NetworkConfig(activation=ActivationType.SIGMOID)),
            ("With Leaky ReLU Activation", NetworkConfig(activation=ActivationType.LEAKY_RELU)),
        ]
        
        trainers = []
        
        for name, net_config in configs:
            print(f"\n🔬 Experiment: {name}")
            print("-" * 40)
            
            # Create and train model
            model = DynamicNeuralNetwork(net_config)
            print(model.get_architecture_summary())
            
            trainer = NeuralTrainer(self.training_config)
            trained_model = trainer.train_model(model, x_train, y_train)
            
            # Store results
            self.results[name] = {
                'model': trained_model,
                'trainer': trainer,
                'config': net_config
            }
            trainers.append(trainer)
            
            # Plot individual results
            self.visualizer.plot_prediction_comparison(
                trained_model, 
                f"Neural Network: {name}"
            )
        
        # Plot training curves comparison
        self.visualizer.plot_training_curves(
            trainers, 
            [name for name, _ in configs]
        )
        
        # Print summary
        self._print_experiment_summary()
    
    def _print_experiment_summary(self):
        print("\n📋 Experiment Summary")
        print("=" * 60)
        
        for name, result in self.results.items():
            final_loss = result['trainer'].get_final_loss()
            activation = result['config'].activation.value
            print(f"🔸 {name:<25} | Activation: {activation:<10} | Final Loss: {final_loss:.6f}")

# Main execution
if __name__ == "__main__":
    # Create custom training configuration
    config = TrainingConfig(
        epochs=1000,
        learning_rate=0.001,
        print_interval=200,
        seed=42
    )
    
    # Run the experiment
    experiment = NeuralNetworkExperiment(config)
    experiment.run_experiment()