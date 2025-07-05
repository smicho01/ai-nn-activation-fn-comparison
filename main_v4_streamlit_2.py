import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, Optional

st.set_page_config(layout="wide")

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
    def get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

class DataGenerator:
    def __init__(self, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = DeviceManager.get_device()

    def generate(self, power=3, x_range=(-10, 10), num_points=1000, noise_std=0.0):
        x = torch.linspace(x_range[0], x_range[1], num_points).unsqueeze(1)
        y = x ** power
        if noise_std > 0:
            y += torch.randn_like(y) * noise_std
        return x.to(self.device), y.to(self.device)

class ActivationFactory:
    @staticmethod
    def get(activation: ActivationType) -> Optional[nn.Module]:
        return {
            ActivationType.NONE: None,
            ActivationType.RELU: nn.ReLU(),
            ActivationType.TANH: nn.Tanh(),
            ActivationType.SIGMOID: nn.Sigmoid(),
            ActivationType.LEAKY_RELU: nn.LeakyReLU(0.1),
            ActivationType.ELU: nn.ELU(),
            ActivationType.GELU: nn.GELU()
        }.get(activation)

class DynamicNet(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        self.model = nn.Sequential(*self._build_layers(config))

    def _build_layers(self, config):
        layers = [nn.Linear(config.input_dim, config.hidden_dim)]
        act = ActivationFactory.get(config.activation)
        if act:
            layers.append(act)
        for _ in range(config.num_layers - 2):
            layers.append(nn.Linear(config.hidden_dim, config.hidden_dim))
            if act:
                layers.append(act)
        layers.append(nn.Linear(config.hidden_dim, config.output_dim))
        return layers

    def forward(self, x):
        return self.model(x)

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = DeviceManager.get_device()
        self.history = []

    def train(self, model, x, y, bar=None, text=None):
        model = model.to(self.device)
        loss_fn = nn.MSELoss()
        opt = optim.Adam(model.parameters(), lr=self.config.learning_rate)
        for epoch in range(self.config.epochs):
            model.train()
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            self.history.append(loss.item())
            if bar:
                bar.progress((epoch + 1) / self.config.epochs)
            if text and (epoch + 1) % 100 == 0:
                text.text(f"Epoch {epoch+1} | Loss: {loss.item():.6f}")
        return model

    def final_loss(self):
        return self.history[-1] if self.history else float('inf')

class Visualizer:
    def __init__(self):
        self.device = DeviceManager.get_device()

    def plot_prediction(self, model, title, x_range=(-10, 10), true_fn=lambda x: x**3, num_points=1000):
        model.eval()
        with torch.no_grad():
            x_vals = torch.linspace(*x_range, num_points).unsqueeze(1).to(self.device)
            y_true = true_fn(x_vals)
            y_pred = model(x_vals)
        x_np, y_t, y_p = x_vals.cpu().numpy().flatten(), y_true.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_np, y=y_t, mode='lines', name='True'))
        fig.add_trace(go.Scatter(x=x_np, y=y_p, mode='lines', name='Predicted'))
        return fig

    def plot_loss_curves(self, trainers, labels):
        fig = go.Figure()
        for t, l in zip(trainers, labels):
            fig.add_trace(go.Scatter(x=list(range(1, len(t.history)+1)), y=t.history, mode='lines', name=l))
        return fig

    def compare_metrics(self, results):
        rows = []
        for name, res in results.items():
            m = res['model']
            t = res['trainer']
            c = res['config']
            rows.append({
                'Model': name,
                'Activation': c.activation.value,
                'Layers': c.num_layers,
                'Hidden Dim': c.hidden_dim,
                'Params': sum(p.numel() for p in m.parameters()),
                'Loss': t.final_loss()
            })
        return pd.DataFrame(rows)

def main():
    st.title("Neural Network Comparison")

    if 'results' not in st.session_state:
        st.session_state.results = {}
    if 'data' not in st.session_state:
        st.session_state.data = None

    st.sidebar.header("Configuration")
    epochs = st.sidebar.slider("Epochs", 100, 5000, 1000, step=100)
    lr = st.sidebar.select_slider("Learning Rate", [0.1, 0.01, 0.001, 0.0001], value=0.001)
    layers = st.sidebar.slider("Layers", 3, 10, 5)
    hidden = st.sidebar.slider("Hidden Units", 16, 256, 64, step=16)
    power = st.sidebar.slider("Polynomial Power", 1, 5, 3)
    x_min = st.sidebar.slider("X Min", -20, 0, -10)
    x_max = st.sidebar.slider("X Max", 0, 20, 10)
    n_points = st.sidebar.slider("Points", 100, 2000, 1000, step=100)
    noise = st.sidebar.slider("Noise Std", 0.0, 5.0, 0.0, step=0.1)
    seed = st.sidebar.slider("Seed", 0, 100, 42)
    acts = st.sidebar.multiselect("Activations", [a.value for a in ActivationType], default=["relu", "tanh", "sigmoid", "leaky_relu", "elu", "gelu"])

    config = TrainingConfig(epochs=epochs, learning_rate=lr, seed=seed)

    if st.button("Run Training"):
        dg = DataGenerator(seed)
        x, y = dg.generate(power, (x_min, x_max), n_points, noise)
        st.session_state.data = (x, y)
        st.session_state.results = {}
        vis = Visualizer()
        for act_str in acts:
            act = ActivationType(act_str)
            net_cfg = NetworkConfig(hidden_dim=hidden, num_layers=layers, activation=act)
            model = DynamicNet(net_cfg)
            trainer = Trainer(config)
            pb = st.progress(0)
            msg = st.empty()
            start = time.time()
            trained = trainer.train(model, x, y, pb, msg)
            elapsed = time.time() - start
            st.session_state.results[act_str] = {
                'model': trained,
                'trainer': trainer,
                'config': net_cfg,
                'time': elapsed
            }
            pb.empty()
            msg.empty()

    if st.session_state.results:
        st.header("Results")
        vis = Visualizer()
        x, y = st.session_state.data

        for act_str, res in st.session_state.results.items():
            fig = vis.plot_prediction(res['model'], f"{act_str.upper()} Activation", (x_min, x_max), lambda x: x**power)
            st.subheader(act_str)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Loss Curves")
        tlist = [r['trainer'] for r in st.session_state.results.values()]
        labels = list(st.session_state.results.keys())
        fig = vis.plot_loss_curves(tlist, labels)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Metrics")
        df = vis.compare_metrics(st.session_state.results)
        st.dataframe(df, use_container_width=True)

main()
