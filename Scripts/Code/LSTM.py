import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===================== PORTABLE PATH CONFIGURATION =====================
# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all paths relative to the script location
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
plots_val_dir = os.path.join(model_dir, "ValPlots")
plots_test_dir = os.path.join(model_dir, "TestPlots")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_val_dir, exist_ok=True)
os.makedirs(plots_test_dir, exist_ok=True)

# ===================== DATA LOADING =====================
# Load data (ALREADY NORMALIZED DATA)
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

# Extract features (T2M and RH2M)
features = ["T2M", "RH2M"]
train_data = train_df[features].values
val_data = val_df[features].values
test_data = test_df[features].values

# ===================== SEQUENCE GENERATION =====================
def create_sequences(data, seq_length, targets):
    X, y = [], []
    for i in range(len(data) - seq_length - max(targets)):
        X.append(data[i:i + seq_length])
        y.append([data[i + seq_length + t, 0] for t in targets] +
                 [data[i + seq_length + t, 1] for t in targets])
    return np.array(X), np.array(y)

# Parameters
seq_length = 24
targets = [1, 3, 6]
X_train, y_train = create_sequences(train_data, seq_length, targets)
X_val, y_val = create_sequences(val_data, seq_length, targets)
X_test, y_test = create_sequences(test_data, seq_length, targets)

# ===================== MODEL ARCHITECTURE =====================
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.1),
    LSTM(32, return_sequences=True),  # Modified to maintain 3D output
    LSTM(16),
    Dense(8, activation='tanh'),
    Dense(6)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Callbacks
callbacks = [
    ModelCheckpoint(os.path.join(model_dir, "best_model.keras"), monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=5, min_lr=0.00001)
]

# Training
history = model.fit(
    X_train, y_train, epochs=5, batch_size=64, 
    validation_data=(X_val, y_val), callbacks=callbacks, verbose=1
)

# ===================== EVALUATION AND REPORTING =====================
def generate_report(y_true, y_pred, horizon_names, file_path):
    report = []
    report.append("COMPLETE METRICS REPORT")
    report.append("=" * 50)
    report.append("\nBY TIME HORIZON:\n")

    # Metrics by horizon
    for i, horizon in enumerate(horizon_names):
        # Calculations for T2M
        t2m_true = y_true[:, i]
        t2m_pred = y_pred[:, i]
        t2m_mse = mean_squared_error(t2m_true, t2m_pred)
        t2m_mae = mean_absolute_error(t2m_true, t2m_pred)
        t2m_mape = np.mean(np.abs((t2m_true - t2m_pred) / t2m_true)) * 100
        t2m_rse = np.sum((t2m_true - t2m_pred) ** 2) / np.sum((t2m_true - np.mean(t2m_true)) ** 2)
        t2m_r2 = r2_score(t2m_true, t2m_pred)

        # Calculations for RH2M
        rh2m_true = y_true[:, i + 3]
        rh2m_pred = y_pred[:, i + 3]
        rh2m_mse = mean_squared_error(rh2m_true, rh2m_pred)
        rh2m_mae = mean_absolute_error(rh2m_true, rh2m_pred)
        rh2m_mape = np.mean(np.abs((rh2m_true - rh2m_pred) / rh2m_true)) * 100
        rh2m_rse = np.sum((rh2m_true - rh2m_pred) ** 2) / np.sum((rh2m_true - np.mean(rh2m_true)) ** 2)
        rh2m_r2 = r2_score(rh2m_true, rh2m_pred)

        report.append(f"T+{horizon}H:")
        report.append(f"  T2M: MSE={t2m_mse:.4f}, MAE={t2m_mae:.4f}, RMSE={np.sqrt(t2m_mse):.4f}, MAPE={t2m_mape:.2f}%, RSE={t2m_rse:.4f}, R²={t2m_r2:.4f}")
        report.append(f"  RH2M: MSE={rh2m_mse:.4f}, MAE={rh2m_mae:.4f}, RMSE={np.sqrt(rh2m_mse):.4f}, MAPE={rh2m_mape:.2f}%, RSE={rh2m_rse:.4f}, R²={rh2m_r2:.4f}\n")

    # Aggregated metrics
    report.append("=" * 50)
    report.append("GENERAL METRICS:\n")

    # For T2M (average of all horizons)
    t2m_all = y_true[:, :3].flatten()
    t2m_pred_all = y_pred[:, :3].flatten()
    t2m_metrics = (
        mean_squared_error(t2m_all, t2m_pred_all),
        mean_absolute_error(t2m_all, t2m_pred_all),
        np.mean(np.abs((t2m_all - t2m_pred_all) / t2m_all)) * 100,
        np.sum((t2m_all - t2m_pred_all) ** 2) / np.sum((t2m_all - np.mean(t2m_all)) ** 2),
        r2_score(t2m_all, t2m_pred_all)
    )

    # For RH2M (average of all horizons)
    rh2m_all = y_true[:, 3:].flatten()
    rh2m_pred_all = y_pred[:, 3:].flatten()
    rh2m_metrics = (
        mean_squared_error(rh2m_all, rh2m_pred_all),
        mean_absolute_error(rh2m_all, rh2m_pred_all),
        np.mean(np.abs((rh2m_all - rh2m_pred_all) / rh2m_all)) * 100,
        np.sum((rh2m_all - rh2m_pred_all) ** 2) / np.sum((rh2m_all - np.mean(rh2m_all)) ** 2),
        r2_score(rh2m_all, rh2m_pred_all)
    )

    # All metrics combined
    all_metrics = (
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        r2_score(y_true, y_pred)
    )

    report.append("T2M:")
    report.append(f"  MSE={t2m_metrics[0]:.4f}, MAE={t2m_metrics[1]:.4f}, RMSE={np.sqrt(t2m_metrics[0]):.4f}, MAPE={t2m_metrics[2]:.2f}%, RSE={t2m_metrics[3]:.4f}, R²={t2m_metrics[4]:.4f}\n")

    report.append("RH2M:")
    report.append(f"  MSE={rh2m_metrics[0]:.4f}, MAE={rh2m_metrics[1]:.4f}, RMSE={np.sqrt(rh2m_metrics[0]):.4f}, MAPE={rh2m_metrics[2]:.2f}%, RSE={rh2m_metrics[3]:.4f}, R²={rh2m_metrics[4]:.4f}\n")

    report.append("ALL:")
    report.append(f"  MSE={all_metrics[0]:.4f}, MAE={all_metrics[1]:.4f}, RMSE={np.sqrt(all_metrics[0]):.4f}, MAPE={all_metrics[2]:.2f}%, RSE={all_metrics[3]:.4f}, R²={all_metrics[4]:.4f}")

    # Save report
    with open(file_path, 'w') as f:
        f.write("\n".join(report))

# Load best model and generate reports
model = keras.models.load_model(os.path.join(model_dir, "best_model.keras"))

# Validation Report
y_val_pred = model.predict(X_val)
generate_report(y_val, y_val_pred, targets, os.path.join(plots_val_dir, "val_metrics.txt"))

# Test Report
y_test_pred = model.predict(X_test)
generate_report(y_test, y_test_pred, targets, os.path.join(plots_test_dir, "test_metrics.txt"))

# ===================== PLOTTING =====================
def save_plots(history, X, y_true, y_pred, plot_dir, prefix):
    # Training plots (only for validation)
    if prefix == "val":
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Train MSE')
        plt.plot(history.history['val_loss'], label='Validation MSE')
        plt.title('Training MSE Evolution')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(plot_dir, f"{prefix}_mse_history.png"))
        plt.close()

    # Comparison plots
    plt.figure(figsize=(18, 12))
    for i, t in enumerate(targets):
        plt.subplot(3, 2, i + 1)
        plt.plot(y_true[:100, i], label='True', color='navy')
        plt.plot(y_pred[:100, i], '--', label='Predicted', color='firebrick')
        plt.title(f'T2M - t+{t}h')
        plt.legend()

        plt.subplot(3, 2, i + 4)
        plt.plot(y_true[:100, i + 3], label='True', color='darkgreen')
        plt.plot(y_pred[:100, i + 3], '--', label='Predicted', color='orange')
        plt.title(f'RH2M - t+{t}h')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{prefix}_comparison.png"))
    plt.close()

# Save plots
save_plots(history, X_val, y_val, y_val_pred, plots_val_dir, "val")
save_plots(history, X_test, y_test, y_test_pred, plots_test_dir, "test")

print("Training and evaluation completed")