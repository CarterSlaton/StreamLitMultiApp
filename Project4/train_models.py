"""
Pre-train all models and save results for instant loading in Streamlit app.
Run this script locally to generate the cached results.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pickle
import os
from datetime import datetime

# =============================================================================
# MODEL DEFINITION (same as in app.py)
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

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    
    return X, y

def prepare_data(df, temperature_col='T (degC)', sequence_length=144, train_ratio=0.8):
    temperature = df[temperature_col].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    temperature_normalized = scaler.fit_transform(temperature)
    
    X, y = create_sequences(temperature_normalized.flatten(), sequence_length)
    
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
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val.to(device))
            val_loss = criterion(val_outputs, y_val.to(device))
            val_losses.append(val_loss.item())
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss.item():.6f}")
    
    training_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'training_time': training_time,
        'num_parameters': sum(p.numel() for p in model.parameters())
    }

def evaluate_model(model, X_test, y_test, scaler, device):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
    
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    predictions_actual = scaler.inverse_transform(predictions)
    
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    mae = mean_absolute_error(y_test_actual, predictions_actual)
    
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
# MAIN TRAINING SCRIPT
# =============================================================================

def main():
    print("=" * 80)
    print("PRE-TRAINING MODELS FOR STREAMLIT APP")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    data_path = 'jena_climate_2009_2016.csv'
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} records")
    
    # Use sample for faster training and less memory
    df_to_use = df.iloc[::20]  # Even smaller sample
    print(f"Using sample: {len(df_to_use):,} records")
    
    # Prepare data
    print("\nPreparing data...")
    prepared_data = prepare_data(df_to_use, sequence_length=144, train_ratio=0.8)
    print(f"Training sequences: {len(prepared_data['X_train']):,}")
    print(f"Test sequences: {len(prepared_data['X_test']):,}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model configurations
    model_configs = [
        {
            'name': 'Simple LSTM',
            'hidden_size': 32,
            'num_layers': 1,
            'dropout': 0.0,
            'color': '#1f77b4',
            'description': 'Basic single-layer LSTM'
        },
        {
            'name': 'Medium LSTM',
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'color': '#ff7f0e',
            'description': '2-layer LSTM with dropout'
        },
        {
            'name': 'Deep LSTM',
            'hidden_size': 128,
            'num_layers': 3,
            'dropout': 0.3,
            'color': '#2ca02c',
            'description': '3-layer deep LSTM network'
        }
    ]
    
    # Train all models
    all_results = []
    
    for idx, config in enumerate(model_configs):
        print("\n" + "=" * 80)
        print(f"Training {config['name']} ({idx+1}/3)")
        print("=" * 80)
        
        # Initialize model
        model = TemperatureRNN(
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout=config['dropout']
        )
        
        # Train
        training_results = train_model(
            model=model,
            X_train=prepared_data['X_train'],
            y_train=prepared_data['y_train'],
            X_val=prepared_data['X_test'],
            y_val=prepared_data['y_test'],
            epochs=10,
            learning_rate=0.001,
            batch_size=64,  # Larger batch = less memory for gradient accumulation
            device=device
        )
        
        # Evaluate
        eval_results = evaluate_model(
            model=model,
            X_test=prepared_data['X_test'],
            y_test=prepared_data['y_test'],
            scaler=prepared_data['scaler'],
            device=device
        )
        
        print(f"\nResults:")
        print(f"  RMSE: {eval_results['rmse']:.4f} C")
        print(f"  MAE: {eval_results['mae']:.4f} C")
        print(f"  R2: {eval_results['r2']:.4f}")
        print(f"  Training time: {training_results['training_time']:.2f}s")
        
        # Store results
        result = {
            'config': config,
            'training': training_results,
            'evaluation': {
                'rmse': float(eval_results['rmse']),
                'mae': float(eval_results['mae']),
                'r2': float(eval_results['r2']),
                'predictions': eval_results['predictions'].tolist(),
                'actual': eval_results['actual'].tolist()
            }
        }
        
        all_results.append(result)
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_file = 'pretrained_results.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({
            'all_results': all_results,
            'prepared_data_info': {
                'train_sequences': len(prepared_data['X_train']),
                'test_sequences': len(prepared_data['X_test']),
                'sequence_length': prepared_data['sequence_length']
            },
            'timestamp': datetime.now().isoformat()
        }, f)
    
    print(f"Results saved to: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / 1024:.2f} KB")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_idx = min(range(len(all_results)), key=lambda i: all_results[i]['evaluation']['rmse'])
    best_model = all_results[best_idx]
    
    print(f"\nBest Model: {best_model['config']['name']}")
    print(f"  RMSE: {best_model['evaluation']['rmse']:.4f} C")
    print(f"  MAE: {best_model['evaluation']['mae']:.4f} C")
    print(f"  R2: {best_model['evaluation']['r2']:.4f}")
    
    print("\nAll results:")
    for result in all_results:
        print(f"  {result['config']['name']:15s} - RMSE: {result['evaluation']['rmse']:.4f} C, R2: {result['evaluation']['r2']:.4f}")
    
    print("\nDone! Use these results in your Streamlit app.")

if __name__ == "__main__":
    main()
