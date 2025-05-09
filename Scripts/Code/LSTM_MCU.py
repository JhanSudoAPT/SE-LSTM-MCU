import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

# ===================== CONFIGURATION =====================
# Portable paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
val_plots_dir = os.path.join(model_dir, "ValPlots")
test_plots_dir = os.path.join(model_dir, "TestPlots")
model_path = os.path.join(model_dir, "best_model.keras")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(val_plots_dir, exist_ok=True)
os.makedirs(test_plots_dir, exist_ok=True)

# Hyperparameters
window_size = 24
pred_steps = [1, 3, 6]

# ===================== FUNCTIONS =====================
def load_data(file_path):
    """Loads UNNORMALIZED data."""
    df = pd.read_csv(file_path)
    return df[['T2M', 'RH2M']].values

def generate_sequences(data):
    """Generates sequences for the model."""
    X_cont, y = [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X_cont.append(data[i - window_size:i])
        y.append(np.hstack([data[i + step] for step in pred_steps]).flatten())
    return np.array(X_cont), np.array(y)

def calculate_mape(y_true, y_pred):
    """Calculates MAPE avoiding division by zero."""
    epsilon = 1e-10  # Small value to avoid division by zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

def calculate_rse(y_true, y_pred):
    """Calculates Relative Squared Error (RSE)."""
    squared_error = np.sum((y_true - y_pred)**2)
    mean_true = np.mean(y_true)
    squared_deviation = np.sum((y_true - mean_true)**2)
    epsilon = 1e-10  # To avoid division by zero
    return squared_error / (squared_deviation + epsilon)

def generate_report(y_true, y_pred, plots_path, report_type='val'):
    """Generates metrics report and plots."""
    metrics = {}

    # Lists to store all metrics
    all_t2m_mse, all_t2m_mae, all_t2m_rmse, all_t2m_mape, all_t2m_r2, all_t2m_rse = [], [], [], [], [], []
    all_rh2m_mse, all_rh2m_mae, all_rh2m_rmse, all_rh2m_mape, all_rh2m_r2, all_rh2m_rse = [], [], [], [], [], []

    for i, t in enumerate(pred_steps):
        # Metrics for T2M
        t2m_true = y_true[:, i]
        t2m_pred = y_pred[:, i]

        # Metrics for RH2M
        rh2m_true = y_true[:, i + 3]
        rh2m_pred = y_pred[:, i + 3]

        # Calculate metrics for T2M
        t2m_metrics = {
            'mse': mean_squared_error(t2m_true, t2m_pred),
            'mae': mean_absolute_error(t2m_true, t2m_pred),
            'rmse': np.sqrt(mean_squared_error(t2m_true, t2m_pred)),
            'mape': calculate_mape(t2m_true, t2m_pred),
            'r2': r2_score(t2m_true, t2m_pred),
            'rse': calculate_rse(t2m_true, t2m_pred)
        }

        # Calculate metrics for RH2M
        rh2m_metrics = {
            'mse': mean_squared_error(rh2m_true, rh2m_pred),
            'mae': mean_absolute_error(rh2m_true, rh2m_pred),
            'rmse': np.sqrt(mean_squared_error(rh2m_true, rh2m_pred)),
            'mape': calculate_mape(rh2m_true, rh2m_pred),
            'r2': r2_score(rh2m_true, rh2m_pred),
            'rse': calculate_rse(rh2m_true, rh2m_pred)
        }

        metrics[f't+{t}h'] = {
            'T2M': t2m_metrics,
            'RH2M': rh2m_metrics
        }

        # Accumulate metrics
        all_t2m_mse.append(t2m_metrics['mse'])
        all_t2m_mae.append(t2m_metrics['mae'])
        all_t2m_rmse.append(t2m_metrics['rmse'])
        all_t2m_mape.append(t2m_metrics['mape'])
        all_t2m_r2.append(t2m_metrics['r2'])
        all_t2m_rse.append(t2m_metrics['rse'])

        all_rh2m_mse.append(rh2m_metrics['mse'])
        all_rh2m_mae.append(rh2m_metrics['mae'])
        all_rh2m_rmse.append(rh2m_metrics['rmse'])
        all_rh2m_mape.append(rh2m_metrics['mape'])
        all_rh2m_r2.append(rh2m_metrics['r2'])
        all_rh2m_rse.append(rh2m_metrics['rse'])

    # Text report
    with open(os.path.join(plots_path, f"{report_type}_report.txt"), "w") as f:
        f.write("METRICS REPORT\n")
        f.write("=" * 50 + "\n")

        for horizon, values in metrics.items():
            f.write(f"{horizon}:\n")
            f.write(f"  T2M - MSE: {values['T2M']['mse']:.4f}, MAE: {values['T2M']['mae']:.4f}, RMSE: {values['T2M']['rmse']:.4f}, "
                    f"MAPE: {values['T2M']['mape']:.2f}%, R²: {values['T2M']['r2']:.4f}, RSE: {values['T2M']['rse']:.4f}\n")
            f.write(f"  RH2M - MSE: {values['RH2M']['mse']:.4f}, MAE: {values['RH2M']['mae']:.4f}, RMSE: {values['RH2M']['rmse']:.4f}, "
                    f"MAPE: {values['RH2M']['mape']:.2f}%, R²: {values['RH2M']['r2']:.4f}, RSE: {values['RH2M']['rse']:.4f}\n\n")

    # Plots
    plt.figure(figsize=(18, 12))
    for i, t in enumerate(pred_steps):
        plt.subplot(2, 3, i + 1)
        plt.plot(y_true[:100, i], label='Actual', color='navy')
        plt.plot(y_pred[:100, i], '--', label=f'Pred (R²={metrics[f"t+{t}h"]["T2M"]["r2"]:.3f})', color='firebrick')
        plt.title(f'T2M - t+{t}h')
        plt.legend()

        plt.subplot(2, 3, i + 4)
        plt.plot(y_true[:100, i + 3], label='Actual', color='darkgreen')
        plt.plot(y_pred[:100, i + 3], '--', label=f'Pred (R²={metrics[f"t+{t}h"]["RH2M"]["r2"]:.3f})', color='orange')
        plt.title(f'RH2M - t+{t}h')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"comparison_{report_type}.png"), dpi=300)
    plt.close()

# ===================== MODEL =====================
def build_model():
    """Builds the LSTM model."""
    input_cont = Input(shape=(window_size, 2))
    x = LSTM(64, activation='relu')(input_cont)
    x = Dense(32, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(len(pred_steps) * 2, activation='linear')(x)
    return Model(inputs=input_cont, outputs=output)

# ===================== TRAINING =====================
def main():
    # Load data
    train_data = load_data(os.path.join(data_dir, "train.csv"))
    val_data = load_data(os.path.join(data_dir, "val.csv"))
    test_data = load_data(os.path.join(data_dir, "test.csv"))

    # Generate sequences
    X_cont_train, y_train = generate_sequences(train_data)
    X_cont_val, y_val = generate_sequences(val_data)
    X_cont_test, y_test = generate_sequences(test_data)

    # Build and train model
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    history = model.fit(
        X_cont_train,
        y_train,
        validation_data=(X_cont_val, y_val),
        epochs=5, ##This model convergues around 40-80 epochs.
        batch_size=32,
        callbacks=[early_stopping]
    )

    # Loss plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train MSE', color='blue')
    plt.plot(history.history['val_loss'], label='Validation MSE', color='red')
    plt.title('MSE Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color='green')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
    plt.title('MAE Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(val_plots_dir, "loss_metrics.png"), dpi=300)
    plt.close()

    # Generate reports
    y_pred_val = model.predict(X_cont_val)
    generate_report(y_val, y_pred_val, val_plots_dir, 'val')

    y_pred_test = model.predict(X_cont_test)
    generate_report(y_test, y_pred_test, test_plots_dir, 'test')

    # Save model
    model.save(model_path)
    print("Training completed. Model and reports saved.")

if __name__ == "__main__":
    main()
