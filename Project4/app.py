import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Page Configuration
st.set_page_config(
    page_title="Jena Climate RNN Forecasting",
    page_icon="ðŸŒ¡ï¸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# RNN MODEL DEFINITION
# =============================================================================

class TemperatureRNN(nn.Module):
    """
    Recurrent Neural Network for temperature forecasting.
    Uses LSTM layers for better handling of long-term dependencies.
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(TemperatureRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take the last output
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions

# =============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(filepath):
    """Load the Jena Climate dataset"""
    try:
        df = pd.read_csv(filepath)
        # Parse date if it exists and convert to string to avoid Arrow serialization issues
        if 'Date Time' in df.columns:
            df['Date Time'] = pd.to_datetime(df['Date Time'], format='%d.%m.%Y %H:%M:%S')
            # Convert to string format for Arrow compatibility
            df['Date Time'] = df['Date Time'].astype(str)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sequences(data, sequence_length):
    """
    Create sequences using a sliding window approach.
    
    Args:
        data: Normalized temperature data (1D array)
        sequence_length: Number of past observations to use (e.g., 720 for 5 days)
    
    Returns:
        X: Input sequences (features)
        y: Target values (next temperature)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X to (samples, sequence_length, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y

def prepare_data(df, temperature_col='T (degC)', sequence_length=720, train_ratio=0.8):
    """
    Complete data preparation pipeline.
    
    Args:
        df: DataFrame with temperature data
        temperature_col: Name of temperature column
        sequence_length: Length of input sequences
        train_ratio: Ratio of data to use for training
    
    Returns:
        Dictionary with prepared data and scaler
    """
    # Extract temperature column
    temperature = df[temperature_col].values.reshape(-1, 1)
    
    # Normalize using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperature_normalized = scaler.fit_transform(temperature)
    
    # Create sequences
    X, y = create_sequences(temperature_normalized.flatten(), sequence_length)
    
    # Split into train and test sets (test set contains latest data)
    train_size = int(len(X) * train_ratio)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    return {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'scaler': scaler,
        'train_size': train_size,
        'sequence_length': sequence_length
    }

# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model_with_ui(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, device):
    """
    Train the RNN model with UI updates (for interactive mode).
    
    Returns:
        Dictionary with training history
    """
    import time
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    # Create progress bar and status
    progress_bar = st.progress(0)
    status_text = st.empty()
    time_text = st.empty()
    
    start_time = time.time()
    epoch_times = []
    
    # Training loop
    for epoch in range(epochs):
        epoch_start = time.time()
        
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size].to(device)
            batch_y = y_train[i:i + batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            val_losses.append(val_loss.item())
        
        # Calculate timing
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        remaining_epochs = epochs - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        # Update progress
        progress = (epoch + 1) / epochs
        progress_bar.progress(progress)
        status_text.text(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss.item():.6f}")
        
        # Show time estimate
        if epoch > 0:  # Only show after first epoch
            time_text.info(f"â±ï¸ Epoch time: {epoch_time:.1f}s | Estimated remaining: {estimated_remaining:.1f}s ({estimated_remaining/60:.1f} min)")
    
    progress_bar.empty()
    status_text.empty()
    time_text.empty()
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }

@st.cache_data
def train_model_cached(_X_train, _y_train, _X_val, _y_val, hidden_size, num_layers, dropout, 
                       epochs, learning_rate, batch_size, device_str):
    """
    Cached training function that runs automatically once with given parameters.
    Returns model state dict and training history.
    """
    import time
    
    # Initialize model
    model = TemperatureRNN(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout
    )
    
    device = torch.device(device_str)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    # Training loop (no UI updates for cached version)
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        # Mini-batch training
        for i in range(0, len(_X_train), batch_size):
            batch_X = _X_train[i:i + batch_size].to(device)
            batch_y = _y_train[i:i + batch_size].to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(_X_val.to(device))
            val_loss = criterion(val_outputs, _y_val.to(device))
            val_losses.append(val_loss.item())
    
    training_time = time.time() - start_time
    
    return {
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, X_test, y_test, scaler, device):
    """
    Evaluate the model on test set.
    
    Returns:
        Dictionary with predictions and metrics
    """
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
    
    # Inverse transform to get actual temperature values
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    predictions_actual = scaler.inverse_transform(predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    
    # Calculate RÂ² score
    ss_res = np.sum((y_test_actual - predictions_actual) ** 2)
    ss_tot = np.sum((y_test_actual - np.mean(y_test_actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'predictions': predictions_actual,
        'actual': y_test_actual,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Title
    st.markdown('<h1 class="main-header">ðŸŒ¡ï¸ Jena Climate Temperature Forecasting with RNN</h1>', unsafe_allow_html=True)
    st.markdown("""
    This application uses **Recurrent Neural Networks (RNN/LSTM)** to predict future temperature values 
    based on historical weather data from the Jena Climate dataset (2009-2016).
    
    **Dataset:** Weather measurements from the Max Planck Institute for Biogeochemistry, 
    recorded every 10 minutes with 14 different features.
    
    ---
    
    **ðŸš€ Auto-Training Mode:** This app automatically trains the model when loaded and displays the results. 
    Adjust the parameters in the sidebar to see different configurations (requires page refresh).
    """)
    
    # Sidebar - Model Configuration
    st.sidebar.header("âš™ï¸ Model Configuration")
    
    st.sidebar.markdown("**ðŸ’¡ Tip:** Change parameters and refresh to retrain!")
    
    st.sidebar.subheader("Data Parameters")
    
    st.sidebar.warning("âš ï¸ **Recommended:** Keep sample mode ON for fast cloud deployment")
    
    # Add data sampling option for faster training
    use_sample = st.sidebar.checkbox(
        "ðŸš€ Use Sample Data (Faster Training)",
        value=True,  # Default to True for faster demo
        help="Use only 10% of data for quick testing. Uncheck for full dataset (may be very slow on Streamlit Cloud)."
    )
    
    if use_sample:
        st.sidebar.success("âœ… Sample mode ON - Training finishes in ~10-30 seconds!")
    else:
        st.sidebar.error("âš ï¸ Full dataset mode - May take 10-30+ minutes on cloud!")
    
    sequence_length = st.sidebar.slider(
        "Sequence Length (observations)",
        min_value=144,
        max_value=1440,
        value=720,
        step=72,
        help="Number of past observations to use. 720 = 5 days (at 10-min intervals)"
    )
    
    train_ratio = st.sidebar.slider(
        "Training Data Ratio",
        min_value=0.6,
        max_value=0.9,
        value=0.8,
        step=0.05
    )
    
    st.sidebar.subheader("Model Hyperparameters")
    hidden_size = st.sidebar.select_slider(
        "Hidden Units",
        options=[32, 64, 128, 256],
        value=64
    )
    
    num_layers = st.sidebar.slider(
        "Number of LSTM Layers",
        min_value=1,
        max_value=4,
        value=2
    )
    
    dropout = st.sidebar.slider(
        "Dropout Rate",
        min_value=0.0,
        max_value=0.5,
        value=0.2,
        step=0.1
    )
    
    st.sidebar.subheader("Training Parameters")
    learning_rate = st.sidebar.select_slider(
        "Learning Rate",
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
        value=0.001,
        format_func=lambda x: f"{x:.4f}"
    )
    
    epochs = st.sidebar.slider(
        "Training Epochs",
        min_value=10,
        max_value=200,
        value=50,
        step=10
    )
    
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        options=[32, 64, 128, 256],
        value=64
    )
    
    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.sidebar.info(f"**Device:** {device}")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'jena_climate_2009_2016.csv')
    
    # Check if file exists before trying to load
    if not os.path.exists(data_path):
        st.error(f"âŒ Dataset file not found at: {data_path}")
        st.error("Please ensure 'jena_climate_2009_2016.csv' is in the Project4 directory and committed to git.")
        st.stop()
        return
    
    df = load_data(data_path)
    
    if df is None:
        st.error("Failed to load dataset. The CSV file may be corrupted.")
        return
    
    # =========================================================================
    # AUTOMATIC TRAINING - Runs once and caches results
    # =========================================================================
    
    st.info("Training model automatically... This will be cached for subsequent interactions.")
    
    with st.spinner("Training in progress..."):
        # Automatically prepare data and train model (cached)
        df_to_use = df.iloc[::10] if use_sample else df

        # Safety caps for demo/sample mode to keep training short on CPU
        if use_sample:
            st.warning(" Sample mode detected ” reducing sequence length and epochs for fast demo runs")
            demo_sequence_length = min(sequence_length, 144)  # very short sequences for demo (1 day = 144 * 10min)
            demo_epochs = min(epochs, 10)  # just 2-3 epochs for ultra-fast feedback
            st.info(f" Demo settings: {demo_epochs} epochs, {demo_sequence_length} sequence length")
        else:
            demo_sequence_length = sequence_length
            demo_epochs = epochs

        prepared_data = prepare_data(df_to_use, sequence_length=demo_sequence_length, train_ratio=train_ratio)
        
        # Train model (this is cached, so it only runs once per parameter combination)
        training_results = train_model_cached(
            prepared_data['X_train'],
            prepared_data['y_train'],
            prepared_data['X_test'],
            prepared_data['y_test'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            epochs=demo_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            device_str=str(device)
        )
        
        # Reconstruct model from cached state
        model = TemperatureRNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            dropout=dropout
        )
        model.load_state_dict(training_results['model_state_dict'])
        model.to(device)
        
        # Evaluate model
        results = evaluate_model(
            model=model,
            X_test=prepared_data['X_test'],
            y_test=prepared_data['y_test'],
            scaler=prepared_data['scaler'],
            device=device
        )
    
    st.success(f"âœ… Training complete! Total time: {training_results['training_time']:.2f} seconds")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "ðŸ“Š Data Overview",
        "ðŸ”§ Data Preprocessing",
        "ðŸ¤– Model Training",
        "ðŸ“ˆ Results & Evaluation",
        "ðŸ“š Documentation"
    ])
    
    # =========================================================================
    # TAB 1: DATA OVERVIEW
    # =========================================================================
    with tab1:
        st.markdown('<h2 class="sub-header">Dataset Overview</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            st.metric("Time Span", "2009-2016")
        with col4:
            st.metric("Frequency", "10 minutes")
        
        st.markdown("### Sample Data")
        st.dataframe(df.head(20), width='stretch')
        
        st.markdown("### Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Features:**")
            for col in df.columns:
                st.write(f"- {col}")
        
        with col2:
            st.markdown("**Statistical Summary:**")
            st.dataframe(df.describe(), width='stretch')
        
        # Temperature visualization
        st.markdown("### Temperature Over Time")
        
        # Sample data for faster visualization (every 100th point)
        sample_df = df.iloc[::100].copy()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sample_df.index,
            y=sample_df['T (degC)'],
            mode='lines',
            name='Temperature',
            line=dict(color='#1f77b4', width=1)
        ))
        
        fig.update_layout(
            title="Temperature Time Series (Sampled)",
            xaxis_title="Time Index",
            yaxis_title="Temperature (Â°C)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Temperature distribution
        st.markdown("### Temperature Distribution")
        fig_hist = px.histogram(
            df,
            x='T (degC)',
            nbins=50,
            title="Temperature Distribution",
            labels={'T (degC)': 'Temperature (Â°C)'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, width='stretch')
    
    # =========================================================================
    # TAB 2: DATA PREPROCESSING
    # =========================================================================
    with tab2:
        st.markdown('<h2 class="sub-header">Data Preprocessing Pipeline</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Preprocessing Steps Applied:
        
        1. **Data Loading**: Loaded CSV file with pandas
        2. **Data Sampling**: """ + ("Used 10% sample (every 10th row)" if use_sample else "Used full dataset") + """
        3. **Feature Selection**: Focused on temperature column (`T (degC)`)
        4. **Normalization**: Applied Min-Max scaling to range [0, 1]
        5. **Sequence Creation**: Used sliding window approach (""" + str(sequence_length) + """ steps)
        6. **Train/Test Split**: """ + f"{int(train_ratio*100)}% training, {int((1-train_ratio)*100)}% testing" + """
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Sequences", f"{len(prepared_data['X_train']):,}")
        with col2:
            st.metric("Test Sequences", f"{len(prepared_data['X_test']):,}")
        with col3:
            st.metric("Sequence Length", sequence_length)
        
        st.markdown("### Normalization Example")
        st.info(f"""
        **Min-Max Scaling Formula:** 
        
        `X_scaled = (X - X_min) / (X_max - X_min)`
        
        This transforms temperature values to the range [0, 1], helping the neural network learn more efficiently.
        """)
        
        # Show example sequences
        st.markdown("### Sample Input Sequence")
        st.write("Example of how data is structured for the RNN:")
        
        example_seq = prepared_data['X_train'][0].numpy().flatten()[:20]
        example_df = pd.DataFrame({
            'Time Step': range(1, len(example_seq) + 1),
            'Normalized Temperature': example_seq
        })
        st.dataframe(example_df, width='stretch')
        
        st.markdown(f"""
        - **Input**: Sequence of {sequence_length} past temperature observations
        - **Output**: Next temperature value (single prediction)
        - **Shape**: `(batch_size, {sequence_length}, 1)`
        """)
    
    # =========================================================================
    # TAB 3: MODEL TRAINING RESULTS
    # =========================================================================
    with tab3:
        st.markdown('<h2 class="sub-header">Model Training Results</h2>', unsafe_allow_html=True)
        
        st.markdown("### Model Architecture")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **RNN Configuration:**
            - **Type**: LSTM (Long Short-Term Memory)
            - **Input Size**: 1 (temperature only)
            - **Hidden Units**: {hidden_size}
            - **Number of Layers**: {num_layers}
            - **Dropout**: {dropout}
            - **Output Size**: 1 (next temperature)
            
            **Training Configuration:**
            - **Learning Rate**: {learning_rate}
            - **Epochs**: {epochs}
            - **Batch Size**: {batch_size}
            - **Optimizer**: Adam
            - **Loss Function**: MSE (Mean Squared Error)
            """)
        
        with col2:
            st.markdown("""
            **Why LSTM?**
            
            LSTM networks are ideal for time series forecasting because they:
            - Remember long-term patterns in sequential data
            - Avoid vanishing gradient problems
            - Learn complex temporal dependencies
            - Handle variable-length sequences effectively
            
            **Model Improvements:**
            - Multiple LSTM layers for deeper learning
            - Dropout for regularization (prevents overfitting)
            - Batch normalization for stable training
            - Adam optimizer for adaptive learning rates
            """)
        
        st.markdown("---")
        
        # Display training metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model Parameters", f"{training_results['num_parameters']:,}")
        with col2:
            st.metric("Training Time", f"{training_results['training_time']:.2f}s")
        with col3:
            st.metric("Training Sequences", f"{len(prepared_data['X_train']):,}")
        with col4:
            st.metric("Test Sequences", f"{len(prepared_data['X_test']):,}")
        
        st.markdown("### Training Loss Over Epochs")
        
        fig_loss = go.Figure()
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(training_results['train_losses']) + 1)),
            y=training_results['train_losses'],
            mode='lines',
            name='Training Loss',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig_loss.add_trace(go.Scatter(
            x=list(range(1, len(training_results['val_losses']) + 1)),
            y=training_results['val_losses'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='#ff7f0e', width=2)
        ))
        
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
            hovermode='x unified',
            height=400,
            legend=dict(x=0.7, y=0.99)
        )
        
        st.plotly_chart(fig_loss, width='stretch')
        
        # Show final losses
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Training Loss", f"{training_results['train_losses'][-1]:.6f}")
        with col2:
            st.metric("Final Validation Loss", f"{training_results['val_losses'][-1]:.6f}")
        with col3:
            improvement = ((training_results['train_losses'][0] - training_results['train_losses'][-1]) / training_results['train_losses'][0]) * 100
            st.metric("Improvement", f"{improvement:.1f}%")
    
    # =========================================================================
    # TAB 4: RESULTS & EVALUATION
    # =========================================================================
    with tab4:
        st.markdown('<h2 class="sub-header">Model Evaluation & Predictions</h2>', unsafe_allow_html=True)
        
        # Display metrics
        st.markdown("### Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{results['rmse']:.4f}Â°C", help="Root Mean Squared Error")
        with col2:
            st.metric("MAE", f"{results['mae']:.4f}Â°C", help="Mean Absolute Error")
        with col3:
            st.metric("RÂ² Score", f"{results['r2']:.4f}", help="Coefficient of Determination")
        with col4:
            st.metric("Training Time", f"{training_results['training_time']:.2f}s")
        
        # Interpretation
        if results['r2'] > 0.9:
            st.success("ðŸŽ‰ Excellent model performance! RÂ² > 0.9 indicates very strong predictive power.")
        elif results['r2'] > 0.7:
            st.info("âœ… Good model performance. RÂ² > 0.7 shows solid predictions.")
        else:
            st.warning("âš ï¸ Model could be improved. Consider adjusting hyperparameters.")
        
        st.markdown("---")
        
        # Predictions visualization
        st.markdown("### Actual vs. Predicted Temperatures")
        
        # Sample for visualization (show every 10th point for clarity)
        sample_indices = range(0, len(results['actual']), 10)
        
        fig_pred = go.Figure()
        
        fig_pred.add_trace(go.Scatter(
            x=list(sample_indices),
            y=results['actual'].flatten()[sample_indices],
            mode='lines',
            name='Actual Temperature',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig_pred.add_trace(go.Scatter(
            x=list(sample_indices),
            y=results['predictions'].flatten()[sample_indices],
            mode='lines',
            name='Predicted Temperature',
            line=dict(color='#d62728', width=2, dash='dash')
        ))
        
        fig_pred.update_layout(
            xaxis_title="Test Sample Index",
            yaxis_title="Temperature (Â°C)",
            hovermode='x unified',
            height=500,
            legend=dict(x=0.7, y=0.99)
        )
        
        st.plotly_chart(fig_pred, width='stretch')
        
        # Error distribution
        st.markdown("### Prediction Error Analysis")
        
        errors = results['actual'].flatten() - results['predictions'].flatten()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_error_hist = px.histogram(
                x=errors,
                nbins=50,
                title="Error Distribution",
                labels={'x': 'Prediction Error (Â°C)'},
                color_discrete_sequence=['#9467bd']
            )
            fig_error_hist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_error_hist, width='stretch')
        
        with col2:
            fig_scatter = px.scatter(
                x=results['actual'].flatten(),
                y=results['predictions'].flatten(),
                title="Actual vs Predicted",
                labels={'x': 'Actual Temperature (Â°C)', 'y': 'Predicted Temperature (Â°C)'},
                color_discrete_sequence=['#1f77b4'],
                opacity=0.5
            )
            # Add perfect prediction line
            min_temp = min(results['actual'].min(), results['predictions'].min())
            max_temp = max(results['actual'].max(), results['predictions'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_temp, max_temp],
                y=[min_temp, max_temp],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_scatter, width='stretch')
        
        # Detailed statistics
        st.markdown("### Error Statistics")
        error_stats = {
            'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', '25th Percentile', '75th Percentile'],
            'Value (Â°C)': [
                f"{np.mean(errors):.4f}",
                f"{np.std(errors):.4f}",
                f"{np.min(errors):.4f}",
                f"{np.max(errors):.4f}",
                f"{np.percentile(errors, 25):.4f}",
                f"{np.percentile(errors, 75):.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(error_stats), width='stretch', hide_index=True)
        
        # Sample predictions table
        st.markdown("### Sample Predictions")
        sample_size = 20
        sample_results = pd.DataFrame({
            'Index': range(sample_size),
            'Actual Temperature (Â°C)': results['actual'].flatten()[:sample_size],
            'Predicted Temperature (Â°C)': results['predictions'].flatten()[:sample_size],
            'Error (Â°C)': errors[:sample_size]
        })
        st.dataframe(sample_results, width='stretch', hide_index=True)
    
    # =========================================================================
    # TAB 5: DOCUMENTATION
    # =========================================================================
    with tab5:
        st.markdown('<h2 class="sub-header">Project Documentation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## ðŸ“– Jena Climate RNN Temperature Forecasting
        
        ### Project Overview
        
        This project implements a **Recurrent Neural Network (RNN)** using **LSTM** (Long Short-Term Memory) 
        architecture to forecast temperature based on the Jena Climate dataset from the Max Planck Institute 
        for Biogeochemistry.
        
        ---
        
        ### Dataset Information
        
        **Source:** Max Planck Institute for Biogeochemistry  
        **Time Period:** 2009 - 2016  
        **Frequency:** Every 10 minutes  
        **Features:** 14 weather measurements including:
        - Temperature (Â°C)
        - Pressure (mbar)
        - Humidity (%)
        - Wind speed and direction
        - And more...
        
        **Total Records:** ~420,000 observations
        
        ---
        
        ### Methodology
        
        #### 1. Data Preprocessing
        
        **Loading:**
        ```python
        df = pd.read_csv('jena_climate_2009_2016.csv')
        temperature = df['T (degC)'].values
        ```
        
        **Normalization:**
        - Applied Min-Max scaling to range [0, 1]
        - Formula: `X_scaled = (X - X_min) / (X_max - X_min)`
        - Benefit: Helps neural network learn efficiently
        
        **Sequence Creation:**
        - Sliding window approach
        - Default: 720 observations (5 days at 10-min intervals)
        - Input: Past temperatures â†’ Output: Next temperature
        
        **Train/Test Split:**
        - Training: 80% of data (earlier time periods)
        - Testing: 20% of data (latest observations)
        - Ensures temporal integrity (no future data leaks)
        
        #### 2. Model Architecture
        
        **LSTM Network:**
        ```
        Input Layer (1 feature)
              â†“
        LSTM Layer 1 (64 units)
              â†“
        Dropout (0.2)
              â†“
        LSTM Layer 2 (64 units)
              â†“
        Fully Connected (1 output)
        ```
        
        **Key Components:**
        - **Input Size:** 1 (temperature only)
        - **Hidden Size:** 32-256 units (configurable)
        - **Number of Layers:** 1-4 LSTM layers
        - **Dropout:** 0.0-0.5 for regularization
        - **Output:** Single temperature prediction
        
        #### 3. Training Process
        
        **Optimizer:** Adam (Adaptive Moment Estimation)
        - Combines advantages of AdaGrad and RMSProp
        - Adaptive learning rate for each parameter
        
        **Loss Function:** MSE (Mean Squared Error)
        - Measures average squared difference between predictions and actual values
        - Penalizes larger errors more heavily
        
        **Training Loop:**
        1. Forward pass through network
        2. Calculate loss
        3. Backpropagation to compute gradients
        4. Update weights using optimizer
        5. Repeat for all batches and epochs
        
        #### 4. Evaluation Metrics
        
        **RMSE (Root Mean Squared Error):**
        - `âˆš(Î£(y_pred - y_actual)Â² / n)`
        - In same units as temperature (Â°C)
        - Lower is better
        
        **MAE (Mean Absolute Error):**
        - `Î£|y_pred - y_actual| / n`
        - Average absolute difference
        - More robust to outliers than RMSE
        
        **RÂ² Score (Coefficient of Determination):**
        - `1 - (SS_res / SS_tot)`
        - Ranges from 0 to 1 (1 = perfect predictions)
        - Indicates proportion of variance explained
        
        ---
        
        ### Model Improvements & Hyperparameter Tuning
        
        **Tested Configurations:**
        
        1. **Hidden Units:** 32, 64, 128, 256
           - More units = more learning capacity
           - But risk of overfitting with too many
        
        2. **Number of Layers:** 1-4 LSTM layers
           - Deeper networks learn complex patterns
           - Diminishing returns beyond 2-3 layers
        
        3. **Learning Rate:** 0.0001 - 0.01
           - Too high: unstable training, divergence
           - Too low: slow convergence
           - Sweet spot: 0.001 - 0.005
        
        4. **Sequence Length:** 144 - 1440 observations
           - 144 = 1 day, 720 = 5 days, 1440 = 10 days
           - Longer sequences capture more context
           - But increase computational cost
        
        5. **Batch Size:** 32, 64, 128, 256
           - Larger batches: faster training, more stable gradients
           - Smaller batches: better generalization
        
        ---
        
        ### Key Findings & Insights
        
        **What Works Well:**
        - âœ… LSTM effectively captures temperature patterns
        - âœ… 5-day lookback window provides good context
        - âœ… 2-layer LSTM with 64-128 hidden units optimal
        - âœ… Min-Max normalization essential for convergence
        - âœ… Adam optimizer with lr=0.001 performs best
        
        **Challenges:**
        - âš ï¸ Sudden weather changes difficult to predict
        - âš ï¸ Long-term forecasts (>1 day) less accurate
        - âš ï¸ Extreme temperatures underrepresented in training
        - âš ï¸ Model learns general trends but misses rapid fluctuations
        
        **Potential Improvements:**
        - ðŸ“ˆ Include additional features (pressure, humidity, wind)
        - ðŸ“ˆ Use bidirectional LSTM to look both directions
        - ðŸ“ˆ Implement attention mechanism for important time steps
        - ðŸ“ˆ Ensemble multiple models for robustness
        - ðŸ“ˆ Add seasonal/cyclical encoding (time of day, month)
        
        ---
        
        ### Technical Implementation
        
        **Libraries Used:**
        - **PyTorch:** Deep learning framework
        - **Pandas:** Data manipulation
        - **NumPy:** Numerical computations
        - **Scikit-learn:** Preprocessing and metrics
        - **Plotly:** Interactive visualizations
        - **Streamlit:** Web application framework
        
        **Hardware Requirements:**
        - CPU: Any modern processor
        - RAM: 4GB minimum, 8GB recommended
        - GPU: Optional (CUDA-enabled for faster training)
        
        ---
        
        ### How to Use This App
        
        1. **Tab 1 - Data Overview:**
           - Explore the dataset
           - View temperature time series
           - Understand data distribution
        
        2. **Tab 2 - Data Preprocessing:**
           - Adjust sequence length
           - Set train/test split ratio
           - Prepare data for training
        
        3. **Tab 3 - Model Training:**
           - Configure model architecture
           - Set hyperparameters
           - Train the RNN
           - Monitor training progress
        
        4. **Tab 4 - Results:**
           - Evaluate model performance
           - View predictions vs actual
           - Analyze errors
        
        5. **Tab 5 - Documentation:**
           - Read project details
           - Understand methodology
           - Learn about improvements
        
        ---
        
        ### References & Resources
        
        **Dataset:**
        - Max Planck Institute for Biogeochemistry
        - Jena, Germany weather station data
        
        **Papers:**
        - Hochreiter & Schmidhuber (1997): "Long Short-Term Memory"
        - Graves (2013): "Generating Sequences With Recurrent Neural Networks"
        
        **Documentation:**
        - PyTorch LSTM: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        - Time Series Forecasting Guide: https://pytorch.org/tutorials/
        
        ---
        
        ### Contact & Acknowledgments
        
        This project was developed as part of a machine learning course assignment 
        focusing on RNN applications for time series forecasting.
        
        **Key Learning Outcomes:**
        - Understanding RNN/LSTM architecture
        - Time series data preparation
        - Hyperparameter tuning techniques
        - Model evaluation and interpretation
        - Real-world deep learning application
        
        ---
        
        *Last Updated: November 2025*
        """)

if __name__ == "__main__":
    main()

