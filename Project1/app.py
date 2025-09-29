import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Set page config
st.set_page_config(page_title="Multi-layer Perceptron Demo", page_icon="�", layout="wide")

# MLP Class Implementation
class MLP:
    """
    Multilayer Perceptron implementation with backpropagation
    """
    def __init__(self, layers, learning_rate=0.01, activation='relu'):
        """
        Initialize MLP
        
        Args:
            layers: List of layer sizes [input_size, hidden1_size, hidden2_size, ..., output_size]
            learning_rate: Learning rate for gradient descent
            activation: Activation function ('relu', 'sigmoid', 'tanh')
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.activation = activation
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Xavier/He initialization
        for i in range(len(layers) - 1):
            if activation == 'relu':
                # He initialization for ReLU
                weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            else:
                # Xavier initialization for sigmoid/tanh
                weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(1.0 / layers[i])
            
            bias = np.zeros((1, layers[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # Store training history
        self.loss_history = []
        self.accuracy_history = []
    
    def _activation_function(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
        elif self.activation == 'tanh':
            return np.tanh(z)
    
    def _activation_derivative(self, z):
        """Derivative of activation function"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activation_function(z)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
    
    def _softmax(self, z):
        """Softmax activation for output layer"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:  # Output layer
                activation = self._softmax(z)
            else:  # Hidden layers
                activation = self._activation_function(z)
            
            activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """Backward propagation to compute gradients"""
        m = X.shape[0]  # Number of samples
        
        # Convert y to one-hot encoding
        y_onehot = np.eye(self.layers[-1])[y.flatten()]
        
        # Initialize gradients
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        # Output layer error (cross-entropy loss derivative)
        dz = activations[-1] - y_onehot
        
        # Backpropagate through layers
        for i in reversed(range(len(self.weights))):
            # Gradients for weights and biases
            dW[i] = (1/m) * np.dot(activations[i].T, dz)
            db[i] = (1/m) * np.sum(dz, axis=0, keepdims=True)
            
            # Error for previous layer (skip if we're at the input layer)
            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self._activation_derivative(z_values[i-1])
        
        return dW, db
    
    def update_parameters(self, dW, db):
        """Update weights and biases using gradients"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def compute_loss(self, y_pred, y_true):
        """Compute cross-entropy loss"""
        m = y_true.shape[0]
        y_onehot = np.eye(self.layers[-1])[y_true.flatten()]
        
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_onehot * np.log(y_pred_clipped)) / m
        
        return loss
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=1000, verbose=True):
        """Train the MLP"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for epoch in range(epochs):
            # Forward propagation
            activations, z_values = self.forward_propagation(X_train)
            
            # Compute loss
            loss = self.compute_loss(activations[-1], y_train)
            self.loss_history.append(loss)
            
            # Compute accuracy
            predictions = np.argmax(activations[-1], axis=1)
            accuracy = np.mean(predictions == y_train.flatten())
            self.accuracy_history.append(accuracy)
            
            # Backward propagation
            dW, db = self.backward_propagation(X_train, y_train, activations, z_values)
            
            # Update parameters
            self.update_parameters(dW, db)
            
            # Update progress
            progress_bar.progress((epoch + 1) / epochs)
            if (epoch + 1) % 50 == 0:
                status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        
        status_text.text("Training completed!")
    
    def predict(self, X):
        """Make predictions"""
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        activations, _ = self.forward_propagation(X)
        return activations[-1]

# Streamlit App
st.title("🧠 Multi-layer Perceptron (MLP) Interactive Demo")
st.caption("Train and visualize a neural network from scratch!")

# Sidebar controls
with st.sidebar:
    st.header("🎛️ Model Configuration")
    
    # Dataset selection
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Breast Cancer", "Custom Synthetic"]
    )
    
    # Architecture
    st.subheader("Network Architecture")
    hidden1_size = st.slider("Hidden Layer 1 Size", 8, 128, 64, step=8)
    hidden2_size = st.slider("Hidden Layer 2 Size", 8, 64, 32, step=8)
    
    # Training parameters
    st.subheader("Training Parameters")
    learning_rate = st.selectbox("Learning Rate", [0.001, 0.01, 0.1], index=1)
    activation = st.selectbox("Activation Function", ["relu", "sigmoid", "tanh"])
    epochs = st.slider("Training Epochs", 100, 2000, 500, step=100)
    
    # Train button
    train_button = st.button("🚀 Train Model", type="primary")

# Main content
if train_button:
    st.header("📊 Dataset Information")
    
    # Load data based on selection
    if dataset_option == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        target_names = data.target_names
        st.write(f"**Dataset**: Wisconsin Breast Cancer")
    else:
        X, y = make_classification(
            n_samples=1000, n_features=20, n_informative=15, 
            n_redundant=5, n_classes=2, random_state=42
        )
        target_names = ["Class 0", "Class 1"]
        st.write(f"**Dataset**: Custom Synthetic Classification")
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Features", X.shape[1])
    with col2:
        st.metric("Samples", X.shape[0])
    with col3:
        st.metric("Classes", len(np.unique(y)))
    
    # Data preprocessing
    st.header("🔄 Data Preprocessing")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Training Set**")
        st.write(f"Shape: {X_train_scaled.shape}")
        st.write(f"Class distribution: {np.bincount(y_train)}")
    with col2:
        st.write("**Test Set**")
        st.write(f"Shape: {X_test_scaled.shape}")
        st.write(f"Class distribution: {np.bincount(y_test)}")
    
    # Model architecture
    st.header("🏗️ Model Architecture")
    input_size = X_train_scaled.shape[1]
    architecture = [input_size, hidden1_size, hidden2_size, 2]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Input Layer", f"{input_size} neurons")
    with col2:
        st.metric("Hidden Layer 1", f"{hidden1_size} neurons")
    with col3:
        st.metric("Hidden Layer 2", f"{hidden2_size} neurons")
    with col4:
        st.metric("Output Layer", "2 neurons")
    
    # Training
    st.header("🎯 Model Training")
    
    # Create and train model
    mlp = MLP(
        layers=architecture,
        learning_rate=learning_rate,
        activation=activation
    )
    
    # Create validation set
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train the model
    mlp.fit(X_train_sub, y_train_sub, X_val, y_val, epochs=epochs, verbose=False)
    
    # Predictions and evaluation
    st.header("📈 Results & Evaluation")
    
    y_pred = mlp.predict(X_test_scaled)
    y_proba = mlp.predict_proba(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test Accuracy", f"{test_accuracy:.4f}")
    with col2:
        st.metric("Final Loss", f"{mlp.loss_history[-1]:.4f}")
    with col3:
        st.metric("Training Accuracy", f"{mlp.accuracy_history[-1]:.4f}")
    
    # Visualizations
    st.header("📊 Training Visualizations")
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    ax1.plot(mlp.loss_history, color='red', linewidth=2)
    ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(mlp.accuracy_history, color='green', linewidth=2)
    ax2.set_title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=target_names, yticklabels=target_names)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('Actual')
    
    # Prediction confidence
    max_probabilities = np.max(y_proba, axis=1)
    ax4.hist(max_probabilities, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Max Probability')
    ax4.set_ylabel('Frequency')
    ax4.axvline(np.mean(max_probabilities), color='red', linestyle='--', 
                label=f'Mean: {np.mean(max_probabilities):.3f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Classification report
    st.header("📋 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=target_names)
    st.text(report)
    
    # Sample predictions
    st.header("🔍 Sample Predictions")
    st.write("Here are some individual predictions to understand model behavior:")
    
    sample_df = []
    for i in range(min(10, len(y_test))):
        sample_df.append({
            "Sample": i+1,
            "True Class": target_names[y_test[i]],
            "Predicted": target_names[y_pred[i]],
            "Confidence": f"{np.max(y_proba[i]):.3f}",
            "Correct": "✅" if y_test[i] == y_pred[i] else "❌"
        })
    
    st.dataframe(sample_df, use_container_width=True)

else:
    st.info("👆 Configure your model parameters in the sidebar and click 'Train Model' to start!")
    
    st.header("🤔 What is a Multi-layer Perceptron?")
    st.write("""
    A Multi-layer Perceptron (MLP) is a type of artificial neural network consisting of:
    
    - **Input Layer**: Receives the feature data
    - **Hidden Layers**: Process the information with weighted connections and activation functions
    - **Output Layer**: Produces the final predictions
    
    **Key Components:**
    - **Weights & Biases**: Parameters that the model learns during training
    - **Activation Functions**: Non-linear functions (ReLU, Sigmoid, Tanh) that add complexity
    - **Backpropagation**: Algorithm used to update weights based on prediction errors
    - **Loss Function**: Measures how far off the predictions are from the true values
    """)
    
    st.header("🎯 How to Use This Demo")
    st.write("""
    1. **Choose a Dataset**: Select between real breast cancer data or synthetic classification data
    2. **Configure Architecture**: Adjust hidden layer sizes to change model complexity
    3. **Set Training Parameters**: Choose learning rate, activation function, and number of epochs
    4. **Train the Model**: Click the train button and watch the model learn in real-time
    5. **Analyze Results**: View training curves, confusion matrix, and prediction confidence
    """)
    
    st.header("💡 Tips for Better Results")
    st.write("""
    - **Learning Rate**: Start with 0.01. If loss doesn't decrease, try 0.001. If training is slow, try 0.1
    - **Hidden Layers**: Larger layers can learn complex patterns but may overfit
    - **Epochs**: More epochs = more training, but watch for overfitting
    - **Activation Functions**: ReLU works well for most cases, try others for comparison
    """)
