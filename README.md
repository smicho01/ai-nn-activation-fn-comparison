# 🧠 Neural Network Activation Function - Comparison Lab

An interactive Streamlit application for comparing different neural network architectures and activation functions. This tool allows you to visualize the impact of various activation functions on neural network performance when learning polynomial functions.

## 🚀 Features

### 🎛️ Interactive Configuration
- **Training Parameters**: Adjustable epochs, learning rates
- **Network Architecture**: Configurable layers and hidden dimensions
- **Data Generation**: Customizable polynomial functions with noise
- **Activation Functions**: Multiple activation function comparisons

### 📊 Visualizations
- **Real-time Training**: Progress bars and live status updates
- **Interactive Plots**: Plotly-powered charts with zoom and hover
- **Prediction Comparisons**: Side-by-side model performance
- **Training Curves**: Loss progression visualization
- **Metrics Dashboard**: Comprehensive performance comparison

### 🔧 Supported Activation Functions
- **None** (Linear layers only)
- **ReLU** (Rectified Linear Unit)
- **Tanh** (Hyperbolic Tangent)
- **Sigmoid** (Logistic Function)
- **Leaky ReLU** (Leaky Rectified Linear Unit)
- **ELU** (Exponential Linear Unit)
- **GELU** (Gaussian Error Linear Unit)

## 📦 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies
```bash
pip install streamlit torch matplotlib numpy pandas plotly
```

### Alternative Installation (with requirements.txt)
```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file with:
```
streamlit>=1.28.0
torch>=2.0.0
matplotlib>=3.7.0
numpy>=1.24.0
pandas>=2.0.0
plotly>=5.15.0
```

## 🎯 Quick Start

1. **Clone or Download** the application file
2. **Save** the code as `neural_network_app.py`
3. **Run** the application:
   ```bash
   streamlit run neural_network_app.py
   ```
4. **Open** your browser to `http://localhost:8501`
5. **Configure** parameters in the sidebar
6. **Click** "🚀 Generate Data & Train Models" to start training
7. **Explore** results in the different tabs

## 📋 Usage Guide

### 1. Configuration (Sidebar)
- **Training Parameters**: Set epochs (100-5000) and learning rate (0.1-0.00001)
- **Network Architecture**: Choose number of layers (3-10) and hidden dimensions (16-256)
- **Data Generation**: Select polynomial power (1-5), input range, and noise level
- **Activation Functions**: Select which activation functions to compare

### 2. Training Process
- Click the training button to generate data and train all selected models
- Watch real-time progress bars and status updates
- All models train automatically with the same data for fair comparison

### 3. Results Analysis
- **🎯 Predictions Tab**: View how well each model learned the target function
- **📊 Training Curves Tab**: Compare loss progression during training
- **📋 Metrics Tab**: Detailed performance metrics and training times

## 🔬 Technical Details

### Architecture
The application uses a modular, object-oriented design with:
- **Configuration Classes**: Dataclasses for training and network parameters
- **Factory Pattern**: For creating different activation functions
- **Device Management**: Automatic CUDA/MPS/CPU detection
- **Session State**: Persistent results across interactions

### Neural Network Structure
- Configurable number of layers (3-10)
- Adjustable hidden dimensions (16-256)
- Dynamic activation function insertion
- Parameter counting and architecture summary

### Data Generation
- Polynomial functions: y = x^power (power 1-5)
- Configurable input ranges
- Optional Gaussian noise injection
- Reproducible results with seed control

## 📊 Performance Metrics

The application tracks and displays:
- **MSE (Mean Squared Error)**: Primary loss metric
- **MAE (Mean Absolute Error)**: Alternative error measurement
- **Training Time**: Execution time for each model
- **Parameter Count**: Total trainable parameters
- **Final Loss**: Converged loss value

## 🎨 Visualization Features

### Interactive Plots
- **Zoom and Pan**: Detailed examination of results
- **Hover Information**: Point-by-point data inspection
- **Legend Toggle**: Show/hide specific models
- **Responsive Design**: Adapts to different screen sizes

### Professional Styling
- Clean, modern interface design
- Consistent color schemes
- Clear typography and spacing
- Intuitive navigation

## 🔧 Customization Options

### Adding New Activation Functions
1. Add to `ActivationType` enum
2. Update `ActivationFactory.create_activation()`
3. Include in sidebar multiselect options

### Extending Data Generation
- Modify `DataGenerator.generate_polynomial_data()`
- Add new mathematical functions
- Implement custom data patterns

### Custom Visualizations
- Extend `StreamlitVisualizer` class
- Add new plot types with Plotly
- Create custom metrics calculations

## 🐛 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **CUDA Issues**: App automatically falls back to CPU if CUDA unavailable
3. **Memory Issues**: Reduce number of points or layers for large datasets
4. **Browser Issues**: Try refreshing or clearing browser cache

### Performance Tips
- Use fewer training points for faster iterations
- Reduce epochs for quick experimentation
- Select fewer activation functions for faster training

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional activation functions
- New visualization types
- Performance optimizations
- UI/UX enhancements
- Documentation improvements

## 📜 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- **PyTorch**: Deep learning framework
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **NumPy/Pandas**: Data manipulation libraries

## 📞 Support

For questions, issues, or suggestions:
- Create an issue in the repository
- Check the troubleshooting section
- Review Streamlit and PyTorch documentation

---

**Built with ❤️ by SevMadeIT using Streamlit, PyTorch, and Plotly**

*Happy experimenting with neural networks! 🚀*