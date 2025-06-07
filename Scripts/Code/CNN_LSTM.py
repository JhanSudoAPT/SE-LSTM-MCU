import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, TimeDistributed, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===================== PATH CONFIGURATION =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
plots_val_dir = os.path.join(model_dir, "ValPlots")
plots_test_dir = os.path.join(model_dir, "TestPlots")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_val_dir, exist_ok=True)
os.makedirs(plots_test_dir, exist_ok=True)


# ===================== DATA LOADING =====================
def load_data(file_name):
    df = pd.read_csv(os.path.join(data_dir, file_name))
    return df[["T2M", "RH2M"]].values


train_data = load_data("train.csv")
val_data = load_data("val.csv")
test_data = load_data("test.csv")


# ===================== SEQUENCE GENERATION =====================
def create_sequences(data, seq_length, targets):
    X, y = [], []
    for i in range(len(data) - seq_length - max(targets)):
        X.append(data[i:i + seq_length])
        y.append([data[i + seq_length + t, 0] for t in targets] +
                 [data[i + seq_length + t, 1] for t in targets])
    return np.array(X), np.array(y)


seq_length = 24  # 24-hour time window
targets = [1, 3, 6]  # Forecasts for 1, 3 and 6 hours

X_train, y_train = create_sequences(train_data, seq_length, targets)
X_val, y_val = create_sequences(val_data, seq_length, targets)
X_test, y_test = create_sequences(test_data, seq_length, targets)

# ===================== CNN-LSTM ARCHITECTURE =====================
model = Sequential([
        Reshape((seq_length, 2), input_shape=(seq_length, 2)),
        Conv1D(32, 16, activation='relu', padding='same'),
        MaxPooling1D(2),
        Conv1D(64, 16, activation='relu', padding='same'),
        MaxPooling1D(2),
        LSTM(64, return_sequences=False, unroll=True, activation='tanh'),
        Dense(32, activation='relu'),
        Dense(6, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mse',
              metrics=['mae'])

# ===================== TRAINING =====================
callbacks = [
    ModelCheckpoint(os.path.join(model_dir, "best_cnn_lstm_model.keras"),
                    monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=0.00001)
]

history = model.fit(
    X_train, y_train,
    epochs=5,  # This model usually converges within 20-30 epochs
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)


# ===================== LOSS PLOTTING =====================
def plot_loss(history, plot_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss (MSE)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(plot_dir, 'training_loss.png'))
    plt.close()


plot_loss(history, plots_val_dir)

# ===================== EVALUATION =====================
def generate_report(y_true, y_pred, horizon_names, file_path):
    report = []
    report.append("COMPLETE METRICS REPORT")
    report.append("=" * 50)
    report.append("\nBY TIME HORIZON:\n")

    # Metrics per horizon
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

    # For T2M
    t2m_all = y_true[:, :3].flatten()
    t2m_pred_all = y_pred[:, :3].flatten()
    t2m_metrics = (
        mean_squared_error(t2m_all, t2m_pred_all),
        mean_absolute_error(t2m_all, t2m_pred_all),
        np.mean(np.abs((t2m_all - t2m_pred_all) / t2m_all)) * 100,
        np.sum((t2m_all - t2m_pred_all) ** 2) / np.sum((t2m_all - np.mean(t2m_all)) ** 2),
        r2_score(t2m_all, t2m_pred_all)
    )

    # For RH2M
    rh2m_all = y_true[:, 3:].flatten()
    rh2m_pred_all = y_pred[:, 3:].flatten()
    rh2m_metrics = (
        mean_squared_error(rh2m_all, rh2m_pred_all),
        mean_absolute_error(rh2m_all, rh2m_pred_all),
        np.mean(np.abs((rh2m_all - rh2m_pred_all) / rh2m_all)) * 100,
        np.sum((rh2m_all - rh2m_pred_all) ** 2) / np.sum((rh2m_all - np.mean(rh2m_all)) ** 2),
        r2_score(rh2m_all, rh2m_pred_all)
    )

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

    with open(file_path, 'w') as f:
        f.write("\n".join(report))


# Test Evaluation
model = keras.models.load_model(os.path.join(model_dir, "best_cnn_lstm_model.keras"))

for dataset, X, y, dir_name in [('val', X_val, y_val, plots_val_dir),
                                ('test', X_test, y_test, plots_test_dir)]:
    y_pred = model.predict(X)
    generate_report(y, y_pred, targets, os.path.join(dir_name, f"{dataset}_metrics.txt"))

    # Plots
    plt.figure(figsize=(18, 12))
    for i, t in enumerate(targets):
        plt.subplot(3, 2, i + 1)
        plt.plot(y[:100, i], label='Actual', color='blue')
        plt.plot(y_pred[:100, i], label='Predicted', color='red', linestyle='--')
        plt.title(f'T2M t+{t}h')
        plt.legend()

        plt.subplot(3, 2, i + 4)
        plt.plot(y[:100, i + 3], label='Actual', color='green')
        plt.plot(y_pred[:100, i + 3], label='Predicted', color='orange', linestyle='--')
        plt.title(f'RH2M t+{t}h')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(dir_name, f"{dataset}_comparison.png"))
    plt.close()

print("Training and evaluation completed")
