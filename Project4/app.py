import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
import plotly.express as px
import os
import pickle

# Page Configuration
st.set_page_config(
    page_title="Jena Climate RNN Forecasting",
    page_icon="🌡️",
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
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# RNN MODEL DEFINITION
# =============================================================================

class TemperatureRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(TemperatureRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        predictions = self.fc(last_output)
        return predictions

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_data(filepath):
    """Load the Jena Climate dataset"""
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def create_sequences(data, sequence_length):
    """Create sequences using sliding window approach"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    
    return X, y

def prepare_data(df, temperature_col='T (degC)', sequence_length=720, train_ratio=0.8):
    """Prepare data for training"""
    temperature = df[temperature_col].values.reshape(-1, 1)
    
    # Apply Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperature_normalized = scaler.fit_transform(temperature)
    
    # Create sequences
    X, y = create_sequences(temperature_normalized.flatten(), sequence_length)
    
    # Split into train and test
    train_size = int(len(X) * train_ratio)
    
    X_train = torch.FloatTensor(X[:train_size])
    y_train = torch.FloatTensor(y[:train_size]).reshape(-1, 1)
    X_test = torch.FloatTensor(X[train_size:])
    y_test = torch.FloatTensor(y[train_size:]).reshape(-1, 1)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'scaler': scaler,
        'train_size': train_size,
        'sequence_length': sequence_length
    }

def train_model(model, X_train, y_train, X_val, y_val, epochs, learning_rate, batch_size, device):
    """Train the RNN model"""
    import time
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    
    train_losses = []
    val_losses = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size].to(device)
            batch_y = y_train[i:i + batch_size].to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
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
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

def evaluate_model(model, X_test, y_test, scaler, device):
    """Evaluate the model on test set"""
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
    
    # Inverse transform to get actual temperature values
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    predictions_actual = scaler.inverse_transform(predictions)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    
    # Calculate R² score
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
    st.markdown('<h1 class="main-header">🌡️ Jena Climate Temperature Forecasting with RNN</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application uses a Recurrent Neural Network (LSTM) to forecast temperature based on the 
    **Jena Climate Dataset** from the Max Planck Institute for Biogeochemistry.
    
    The dataset contains weather measurements recorded every 10 minutes from 2009 to 2016.
    """)
    
    # Sidebar
    st.sidebar.header("📊 Pre-Trained Models")
    st.sidebar.info("""
    **Instant Results!**
    
    This app loads pre-trained model results for instant viewing.
    
    Three LSTM configurations:
    - 🔷 Simple LSTM (32 units, 1 layer)
    - 🔶 Medium LSTM (64 units, 2 layers)
    - 🟢 Deep LSTM (128 units, 3 layers)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Training Configuration")
    st.sidebar.write("**Data:** 5% sample (21K records)")
    st.sidebar.write("**Sequence Length:** 144 steps (1 day)")
    st.sidebar.write("**Train/Test Split:** 80/20")
    st.sidebar.write("**Epochs:** 10")
    st.sidebar.write("**Learning Rate:** 0.001")
    st.sidebar.write("**Batch Size:** 64")
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), 'jena_climate_2009_2016.csv')
    
    if not os.path.exists(data_path):
        st.error(f"❌ Dataset file not found at: {data_path}")
        st.error("Please ensure 'jena_climate_2009_2016.csv' is in the Project4 directory.")
        st.stop()
        return
    
    df = load_data(data_path)
    
    if df is None:
        st.error("Failed to load dataset.")
        return
    
    # Check for pre-trained results or train new models
    results_path = os.path.join(os.path.dirname(__file__), 'pretrained_results.pkl')
    
    if os.path.exists(results_path):
        st.info("📥 Loading pre-trained model results...")
        with open(results_path, 'rb') as f:
            pretrained_data = pickle.load(f)
        all_results = pretrained_data['all_results']
        prepared_data_info = pretrained_data['prepared_data_info']
        st.success(f"✅ Loaded {len(all_results)} pre-trained models!")
    else:
        st.warning("⚠️ No pre-trained results found. Training models now...")
        
        # Prepare data
        st.info("Preparing data...")
        df_sample = df.iloc[::20]  # Use 5% sample for faster training
        prepared_data = prepare_data(df_sample, sequence_length=144, train_ratio=0.8)
        
        prepared_data_info = {
            'train_sequences': len(prepared_data['X_train']),
            'test_sequences': len(prepared_data['X_test']),
            'sequence_length': prepared_data['sequence_length']
        }
        
        # Train 3 models
        all_results = []
        model_configs = [
            {'name': 'Simple LSTM', 'hidden_size': 32, 'num_layers': 1, 'dropout': 0.0, 'color': '#1f77b4'},
            {'name': 'Medium LSTM', 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'color': '#ff7f0e'},
            {'name': 'Deep LSTM', 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.3, 'color': '#2ca02c'}
        ]
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for idx, config in enumerate(model_configs):
            status_text.text(f"Training {config['name']} ({idx+1}/3)...")
            
            # Initialize and train model
            model = TemperatureRNN(
                input_size=1,
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=1,
                dropout=config['dropout']
            )
            
            training_results = train_model(
                model, prepared_data['X_train'], prepared_data['y_train'],
                prepared_data['X_test'], prepared_data['y_test'],
                epochs=10, learning_rate=0.001, batch_size=64, device=device
            )
            
            # Evaluate
            eval_results = evaluate_model(
                model, prepared_data['X_test'], prepared_data['y_test'],
                prepared_data['scaler'], device
            )
            
            all_results.append({
                'config': config,
                'training': training_results,
                'evaluation': {
                    'rmse': float(eval_results['rmse']),
                    'mae': float(eval_results['mae']),
                    'r2': float(eval_results['r2']),
                    'predictions': eval_results['predictions'].tolist(),
                    'actual': eval_results['actual'].tolist()
                }
            })
            
            progress_bar.progress((idx + 1) / len(model_configs))
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ All 3 models trained successfully!")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Overview",
        "🔧 Preprocessing",
        "📈 Model Comparison",
        "🎯 Predictions",
        "📖 Documentation"
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
        st.dataframe(df.head(20), use_container_width=True)
        
        st.markdown("### Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Available Features:**")
            for col in df.columns:
                st.write(f"- {col}")
        
        with col2:
            st.markdown("**Statistical Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Temperature visualization
        st.markdown("### Temperature Over Time")
        
        # Sample data for faster visualization
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
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Temperature distribution
        st.markdown("### Temperature Distribution")
        fig_hist = px.histogram(
            df,
            x='T (degC)',
            nbins=50,
            title="Temperature Distribution",
            labels={'T (degC)': 'Temperature (°C)'},
            color_discrete_sequence=['#1f77b4']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # =========================================================================
    # TAB 2: DATA PREPROCESSING
    # =========================================================================
    with tab2:
        st.markdown('<h2 class="sub-header">Data Preprocessing Pipeline</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### Preprocessing Steps Applied:
        
        1. **Data Loading**: Loaded CSV file with pandas
        2. **Data Sampling**: Used 5% sample (every 20th row) - 21K records
        3. **Feature Selection**: Focused on temperature column (`T (degC)`)
        4. **Normalization**: Applied Min-Max scaling to range [0, 1]
        5. **Sequence Creation**: Used sliding window approach (144 steps = 1 day)
        6. **Train/Test Split**: 80% training, 20% testing
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Training Sequences", f"{prepared_data_info['train_sequences']:,}")
        with col2:
            st.metric("Test Sequences", f"{prepared_data_info['test_sequences']:,}")
        with col3:
            st.metric("Sequence Length", prepared_data_info['sequence_length'])
        
        st.markdown("### Normalization Example")
        st.info("""
        **Min-Max Scaling Formula:** 
        
        `X_scaled = (X - X_min) / (X_max - X_min)`
        
        This transforms temperature values to the range [0, 1], helping the neural network learn more efficiently.
        """)
        
        st.markdown("### Sequence Structure")
        st.markdown(f"""
        - **Input**: Sequence of {prepared_data_info['sequence_length']} past temperature observations
        - **Output**: Next temperature value (single prediction)
        - **Example**: Use last 144 observations (1 day at 10-min intervals) to predict next temperature
        """)
    
    # =========================================================================
    # TAB 3: MODEL COMPARISON
    # =========================================================================
    with tab3:
        st.markdown('<h2 class="sub-header">Model Comparison: 3 LSTM Configurations</h2>', unsafe_allow_html=True)
        
        st.markdown("### Performance Metrics")
        
        # Create comparison table
        comparison_data = []
        for result in all_results:
            comparison_data.append({
                'Model': result['config']['name'],
                'Hidden Size': result['config']['hidden_size'],
                'Layers': result['config']['num_layers'],
                'Dropout': result['config']['dropout'],
                'Parameters': result['training']['num_parameters'],
                'Training Time (s)': f"{result['training']['training_time']:.2f}",
                'RMSE (°C)': f"{result['evaluation']['rmse']:.4f}",
                'MAE (°C)': f"{result['evaluation']['mae']:.4f}",
                'R² Score': f"{result['evaluation']['r2']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Highlight best model (lowest RMSE)
        best_idx = min(range(len(all_results)), key=lambda i: all_results[i]['evaluation']['rmse'])
        
        st.dataframe(comparison_df, use_container_width=True)
        st.success(f"🏆 Best Model: **{all_results[best_idx]['config']['name']}** (Lowest RMSE)")
        
        # Training loss curves
        st.markdown("### Training Loss Curves")
        
        fig_loss = go.Figure()
        
        for result in all_results:
            config = result['config']
            training = result['training']
            
            # Training loss
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(training['train_losses']) + 1)),
                y=training['train_losses'],
                mode='lines',
                name=f"{config['name']} - Train",
                line=dict(color=config['color'], width=2)
            ))
            
            # Validation loss
            fig_loss.add_trace(go.Scatter(
                x=list(range(1, len(training['val_losses']) + 1)),
                y=training['val_losses'],
                mode='lines',
                name=f"{config['name']} - Val",
                line=dict(color=config['color'], width=2, dash='dash')
            ))
        
        fig_loss.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Loss (MSE)",
            hovermode='x unified',
            height=500,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)')
        )
        
        st.plotly_chart(fig_loss, use_container_width=True)
        
        st.markdown("""
        **Observations:**
        - Solid lines show training loss
        - Dashed lines show validation loss
        - Lower loss values indicate better performance
        """)
    
    # =========================================================================
    # TAB 4: DETAILED PREDICTIONS
    # =========================================================================
    with tab4:
        st.markdown('<h2 class="sub-header">Prediction Visualizations</h2>', unsafe_allow_html=True)
        
        # Model selector
        selected_model = st.selectbox(
            "Select Model to View:",
            options=[r['config']['name'] for r in all_results],
            index=best_idx
        )
        
        # Get selected model results
        selected_result = next(r for r in all_results if r['config']['name'] == selected_model)
        results = selected_result['evaluation'].copy()
        config = selected_result['config']
        training = selected_result['training']
        
        # Convert lists to numpy arrays
        results['actual'] = np.array(results['actual'])
        results['predictions'] = np.array(results['predictions'])
        
        # Display metrics
        st.markdown(f"### {selected_model} Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RMSE", f"{results['rmse']:.4f}°C")
        with col2:
            st.metric("MAE", f"{results['mae']:.4f}°C")
        with col3:
            st.metric("R² Score", f"{results['r2']:.4f}")
        with col4:
            st.metric("Training Time", f"{training['training_time']:.2f}s")
        
        # Interpretation
        if results['r2'] > 0.9:
            st.success("🎉 Excellent model performance! R² > 0.9")
        elif results['r2'] > 0.7:
            st.info("👍 Good model performance. R² > 0.7")
        else:
            st.warning("⚠️ Model could be improved.")
        
        st.markdown("---")
        
        # Predictions visualization
        st.markdown("### Actual vs. Predicted Temperatures")
        
        # Sample for visualization
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
            name=f'{selected_model} Prediction',
            line=dict(color=config['color'], width=2, dash='dash')
        ))
        
        fig_pred.update_layout(
            xaxis_title="Test Sample Index",
            yaxis_title="Temperature (°C)",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Error distribution
        st.markdown("### Prediction Error Analysis")
        
        errors = results['actual'].flatten() - results['predictions'].flatten()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_error_hist = px.histogram(
                x=errors,
                nbins=50,
                title="Error Distribution",
                labels={'x': 'Prediction Error (°C)'},
                color_discrete_sequence=['#9467bd']
            )
            fig_error_hist.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_error_hist, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(
                x=results['actual'].flatten(),
                y=results['predictions'].flatten(),
                title="Actual vs Predicted",
                labels={'x': 'Actual Temperature (°C)', 'y': 'Predicted Temperature (°C)'},
                color_discrete_sequence=['#1f77b4'],
                opacity=0.5
            )
            # Perfect prediction line
            min_temp = min(results['actual'].min(), results['predictions'].min())
            max_temp = max(results['actual'].max(), results['predictions'].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_temp, max_temp],
                y=[min_temp, max_temp],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Error statistics
        st.markdown("### Error Statistics")
        error_stats = {
            'Metric': ['Mean Error', 'Std Error', 'Min Error', 'Max Error', '25th Percentile', '75th Percentile'],
            'Value (°C)': [
                f"{np.mean(errors):.4f}",
                f"{np.std(errors):.4f}",
                f"{np.min(errors):.4f}",
                f"{np.max(errors):.4f}",
                f"{np.percentile(errors, 25):.4f}",
                f"{np.percentile(errors, 75):.4f}"
            ]
        }
        st.dataframe(pd.DataFrame(error_stats), use_container_width=True, hide_index=True)
    
    # =========================================================================
    # TAB 5: DOCUMENTATION
    # =========================================================================
    with tab5:
        st.markdown('<h2 class="sub-header">Assignment Documentation</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ## Assignment Requirements Completed ✅
        
        ### 1. Data Preprocessing
        - ✅ **Data Loading**: Loaded Jena Climate CSV with pandas
        - ✅ **Normalization**: Applied Min-Max scaling to temperature column
        - ✅ **Sequence Creation**: Used sliding window approach (144 observations = 1 day)
        
        ### 2. Model Adaptation
        - ✅ **Input/Output Adjustment**: RNN adapted for temperature forecasting
        - ✅ **Hyperparameter Tuning**: Three configurations tested:
          - Simple LSTM: 32 hidden units, 1 layer
          - Medium LSTM: 64 hidden units, 2 layers, 0.2 dropout
          - Deep LSTM: 128 hidden units, 3 layers, 0.3 dropout
        
        ### 3. Model Training and Testing
        - ✅ **Data Splitting**: 80% training, 20% testing (latest data for testing)
        - ✅ **Training**: Models trained for 10 epochs with learning rate 0.001
        - ✅ **Evaluation**: Used RMSE, MAE, and R² metrics
        - ✅ **Model Revision**: Three architectures compared for best performance
        
        ### 4. Visualization
        - ✅ **Loss Plotting**: Training and validation loss over epochs
        - ✅ **Prediction Plotting**: Actual vs. predicted temperatures
        - ✅ **Streamlit App**: Complete interactive dashboard
        
        ## Technical Details
        
        ### Dataset
        - **Source**: Jena Climate Dataset (Max Planck Institute)
        - **Records**: 420,551 observations
        - **Frequency**: Every 10 minutes (2009-2016)
        - **Features**: 14 weather measurements
        - **Focus**: Temperature (`T (degC)`)
        
        ### Model Architecture
        - **Type**: LSTM (Long Short-Term Memory) RNN
        - **Input**: Sequences of 144 past temperature values
        - **Output**: Next temperature prediction
        - **Framework**: PyTorch 2.0+
        
        ### Performance Metrics
        - **RMSE**: Root Mean Squared Error (lower is better)
        - **MAE**: Mean Absolute Error (lower is better)
        - **R²**: Coefficient of Determination (closer to 1 is better)
        
        ### Deployment
        - **Platform**: Streamlit Cloud
        - **Strategy**: Pre-trained models for instant loading
        - **Load Time**: < 2 seconds
        
        ## How to Use This App
        
        1. **Data Overview**: Explore the dataset statistics and visualizations
        2. **Preprocessing**: Understand how data was prepared
        3. **Model Comparison**: Compare performance of all three models
        4. **Predictions**: View detailed predictions and error analysis
        5. **Documentation**: Read about the implementation (you are here!)
        
        ## References
        
        - Dataset: Jena Climate Dataset, Max Planck Institute for Biogeochemistry
        - Framework: PyTorch (https://pytorch.org)
        - Visualization: Plotly (https://plotly.com)
        - Deployment: Streamlit (https://streamlit.io)
        """)

if __name__ == "__main__":
    main()
