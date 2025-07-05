import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Set page config
st.set_page_config(
    page_title="Neural Network Comparison Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration and utilities
class ActivationType(Enum):
    NONE = "none"
    RELU = "relu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    GELU = "gelu"

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
            ActivationType.LEAKY_RELU: nn.LeakyReLU(0.1),
            ActivationType.ELU: nn.ELU(),
            ActivationType.GELU: nn.GELU()
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
        return f"Architecture: {self.config.activation.value} | Layers: {self.config.num_layers} | Params: {total_params:,}"

class StreamlitNeuralTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = DeviceManager.get_optimal_device()
        self.loss_history = []
    
    def train_model(self, 
                   model: nn.Module, 
                   x_train: torch.Tensor, 
                   y_train: torch.Tensor,
                   progress_bar=None,
                   status_text=None) -> nn.Module:
        
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        self.loss_history = []
        
        for epoch in range(self.config.epochs):
            model.train()
            optimizer.zero_grad()
            
            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            
            self.loss_history.append(loss.item())
            
            # Update progress bar
            if progress_bar:
                progress_bar.progress((epoch + 1) / self.config.epochs)
            
            if status_text and (epoch + 1) % self.config.print_interval == 0:
                status_text.text(f"Epoch {epoch+1}/{self.config.epochs} | Loss: {loss.item():.6f}")
        
        return model
    
    def get_final_loss(self) -> float:
        return self.loss_history[-1] if self.loss_history else float('inf')

class StreamlitVisualizer:
    def __init__(self):
        self.device = DeviceManager.get_optimal_device()
    
    def create_prediction_plot(self, 
                             model: nn.Module, 
                             title: str,
                             x_range: Tuple[float, float] = (-10, 10),
                             true_function: Callable = lambda x: x**3,
                             num_points: int = 1000) -> go.Figure:
        
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
        mae = np.mean(np.abs(y_true_np - y_pred_np))
        
        fig = go.Figure()
        
        # True function
        fig.add_trace(go.Scatter(
            x=x_np, y=y_true_np,
            mode='lines',
            name='True Function (x³)',
            line=dict(color='navy', width=3)
        ))
        
        # Predicted function
        fig.add_trace(go.Scatter(
            x=x_np, y=y_pred_np,
            mode='lines',
            name='Neural Network Prediction',
            line=dict(color='crimson', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{title}<br><sub>MSE: {mse:.6f} | MAE: {mae:.6f}</sub>',
            xaxis_title='Input (x)',
            yaxis_title='Output (y)',
            template='plotly_white',
            width=800,
            height=500
        )
        
        return fig
    
    def create_training_curves_plot(self, trainers: list, labels: list) -> go.Figure:
        fig = go.Figure()
        
        for trainer, label in zip(trainers, labels):
            epochs = list(range(1, len(trainer.loss_history) + 1))
            fig.add_trace(go.Scatter(
                x=epochs, y=trainer.loss_history,
                mode='lines',
                name=f'{label} (Final: {trainer.get_final_loss():.6f})',
                line=dict(width=3)
            ))
        
        fig.update_layout(
            title='Training Loss Comparison',
            xaxis_title='Epoch',
            yaxis_title='Loss (MSE)',
            yaxis_type='log',
            template='plotly_white',
            width=800,
            height=500
        )
        
        return fig
    
    def create_metrics_comparison(self, results: dict) -> pd.DataFrame:
        metrics_data = []
        
        for name, result in results.items():
            model = result['model']
            trainer = result['trainer']
            config = result['config']
            
            total_params = sum(p.numel() for p in model.parameters())
            final_loss = trainer.get_final_loss()
            
            metrics_data.append({
                'Model': name,
                'Activation': config.activation.value,
                'Parameters': total_params,
                'Final Loss': final_loss,
                'Layers': config.num_layers,
                'Hidden Dim': config.hidden_dim
            })
        
        return pd.DataFrame(metrics_data)

def main():
    st.title("🧠 Neural Network Activation Function - Comparison Lab")
    st.markdown("Compare different neural network architectures and activation functions")
    
    # Initialize session state
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'training_data' not in st.session_state:
        st.session_state.training_data = None
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    # Training parameters
    st.sidebar.subheader("Training Parameters")
    epochs = st.sidebar.slider("Epochs", 100, 5000, 1000, step=100)
    learning_rate = st.sidebar.select_slider("Learning Rate", 
                                            options=[0.1, 0.01, 0.001, 0.0001, 0.00001],
                                            value=0.001)
    
    # Network architecture
    st.sidebar.subheader("Network Architecture")
    num_layers = st.sidebar.slider("Number of Layers", 3, 10, 5)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 16, 256, 64, step=16)
    
    # Data generation
    st.sidebar.subheader("Data Generation")
    polynomial_power = st.sidebar.slider("Polynomial Power", 1, 5, 3)
    x_min = st.sidebar.slider("X Range Min", -20, 0, -10)
    x_max = st.sidebar.slider("X Range Max", 0, 20, 10)
    num_points = st.sidebar.slider("Number of Points", 100, 2000, 1000, step=100)
    noise_std = st.sidebar.slider("Noise Standard Deviation", 0.0, 5.0, 0.0, step=0.1)
    seed = st.sidebar.slider("Random Seed", 0, 100, 42)
    
    # Activation function selection
    st.sidebar.subheader("Activation Functions")
    selected_activations = st.sidebar.multiselect(
        "Select activation functions to compare:",
        options=[act.value for act in ActivationType],
        default=["none", "relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu"]
    )
    
    # Create configurations
    training_config = TrainingConfig(
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎯 Training")
        
        if st.button("🚀 Generate Data & Train Models", type="primary"):
            # Generate data
            data_generator = DataGenerator(seed)
            x_train, y_train = data_generator.generate_polynomial_data(
                power=polynomial_power,
                x_range=(x_min, x_max),
                num_points=num_points,
                noise_std=noise_std
            )
            st.session_state.training_data = (x_train, y_train)
            
            # Clear previous results
            st.session_state.trained_models = {}
            
            # Train models
            visualizer = StreamlitVisualizer()
            
            for activation_str in selected_activations:
                activation_type = ActivationType(activation_str)
                
                st.write(f"Training model with {activation_str} activation...")
                
                # Create progress bar and status text
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create network configuration
                net_config = NetworkConfig(
                    num_layers=num_layers,
                    hidden_dim=hidden_dim,
                    activation=activation_type
                )
                
                # Create and train model
                model = DynamicNeuralNetwork(net_config)
                trainer = StreamlitNeuralTrainer(training_config)
                
                start_time = time.time()
                trained_model = trainer.train_model(
                    model, x_train, y_train, 
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                training_time = time.time() - start_time
                
                # Store results
                st.session_state.trained_models[activation_str] = {
                    'model': trained_model,
                    'trainer': trainer,
                    'config': net_config,
                    'training_time': training_time
                }
                
                status_text.text(f"✅ Training completed in {training_time:.2f}s")
                progress_bar.empty()
                status_text.empty()
            
            st.success("🎉 All models trained successfully!")
    
    with col2:
        st.header("📊 Device Info")
        device = DeviceManager.get_optimal_device()
        st.info(f"Using device: **{device}**")
        
        if st.session_state.training_data is not None:
            x_train, y_train = st.session_state.training_data
            st.metric("Training Samples", len(x_train))
            st.metric("Data Range", f"[{x_min}, {x_max}]")
            st.metric("Polynomial Power", polynomial_power)
    
    # Results section
    if st.session_state.trained_models:
        st.header("📈 Results")
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📊 Training Curves", "📋 Metrics"])
        
        with tab1:
            st.subheader("Model Predictions")
            
            visualizer = StreamlitVisualizer()
            x_train, y_train = st.session_state.training_data
            
            for activation_str, result in st.session_state.trained_models.items():
                model = result['model']
                
                # Create prediction plot
                fig = visualizer.create_prediction_plot(
                    model, 
                    f"Neural Network with {activation_str.upper()} Activation",
                    x_range=(x_min, x_max),
                    true_function=lambda x: x**polynomial_power
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.subheader("Training Loss Curves")
            
            trainers = [result['trainer'] for result in st.session_state.trained_models.values()]
            labels = list(st.session_state.trained_models.keys())
            
            fig = visualizer.create_training_curves_plot(trainers, labels)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Model Comparison Metrics")
            
            metrics_df = visualizer.create_metrics_comparison(st.session_state.trained_models)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Add training time information
            st.subheader("Training Performance")
            perf_data = []
            for activation_str, result in st.session_state.trained_models.items():
                perf_data.append({
                    'Activation': activation_str,
                    'Training Time (s)': f"{result['training_time']:.2f}",
                    'Final Loss': f"{result['trainer'].get_final_loss():.6f}"
                })
            
            perf_df = pd.DataFrame(perf_data)
            st.dataframe(perf_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("🔬 Built with Streamlit, PyTorch, and Plotly")

if __name__ == "__main__":
    main()