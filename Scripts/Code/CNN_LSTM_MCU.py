import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ===================== PATH CONFIGURATION =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories
model_dir = os.path.join(BASE_DIR, "models")
data_dir = os.path.join(BASE_DIR, "data")
val_plots_dir = os.path.join(model_dir, "ValPlots")
test_plots_dir = os.path.join(model_dir, "TestPlots")
model_path = os.path.join(model_dir, "best_model.keras")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(val_plots_dir, exist_ok=True)
os.makedirs(test_plots_dir, exist_ok=True)

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
    max_offset = seq_length + max(targets)
    for i in range(len(data) - max_offset):
        X.append(data[i:i + seq_length])
        y.append([data[i + seq_length + t, 0] for t in targets] +
                 [data[i + seq_length + t, 1] for t in targets])
    return np.array(X), np.array(y)

seq_length = 24
targets = [1, 3, 6]

X_train, y_train = create_sequences(train_data, seq_length, targets)
X_val, y_val = create_sequences(val_data, seq_length, targets)
X_test, y_test = create_sequences(test_data, seq_length, targets)

# ===================== MODEL ARCHITECTURE =====================
def build_model():
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

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model
model = build_model()

# ===================== TRAINING =====================
callbacks = [
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=90,  # This model usually converges between 20-40 epochs.
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ===================== METRIC FUNCTIONS =====================
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100

def calculate_rse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def generate_report(y_true, y_pred, targets, plots_dir, report_type):
    t2m_metrics = {f't+{t}h': {} for t in targets}
    rh2m_metrics = {f't+{t}h': {} for t in targets}

    for i, t in enumerate(targets):
        # Calculations for T2M
        t2m_true = y_true[:, i]
        t2m_pred = y_pred[:, i]
        t2m_metrics[f't+{t}h'] = {
            'mse': mean_squared_error(t2m_true, t2m_pred),
            'mae': mean_absolute_error(t2m_true, t2m_pred),
            'rmse': np.sqrt(mean_squared_error(t2m_true, t2m_pred)),
            'mape': calculate_mape(t2m_true, t2m_pred),
            'rse': calculate_rse(t2m_true, t2m_pred),
            'r2': r2_score(t2m_true, t2m_pred)
        }

        # Calculations for RH2M
        rh2m_true = y_true[:, i + 3]
        rh2m_pred = y_pred[:, i + 3]
        rh2m_metrics[f't+{t}h'] = {
            'mse': mean_squared_error(rh2m_true, rh2m_pred),
            'mae': mean_absolute_error(rh2m_true, rh2m_pred),
            'rmse': np.sqrt(mean_squared_error(rh2m_true, rh2m_pred)),
            'mape': calculate_mape(rh2m_true, rh2m_pred),
            'rse': calculate_rse(rh2m_true, rh2m_pred),
            'r2': r2_score(rh2m_true, rh2m_pred)
        }

    # Averages
    def calculate_averages(metric_dict):
        return {metric: np.mean([values[metric] for values in metric_dict.values()])
                for metric in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']}

    avg_t2m = calculate_averages(t2m_metrics)
    avg_rh2m = calculate_averages(rh2m_metrics)
    avg_general = {metric: np.mean([avg_t2m[metric], avg_rh2m[metric]])
                   for metric in avg_t2m}

    report_path = os.path.join(plots_dir, f"report_{report_type}.txt")
    with open(report_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"METRICS REPORT ({report_type.upper()})\n")
        f.write("=" * 50 + "\n\n")

        f.write("[METRICS BY HORIZON]\n")
        for t in targets:
            f.write(f"\n* t+{t}h:\n")
            f.write(f"  T2M  - MSE: {t2m_metrics[f't+{t}h']['mse']:.4f}, MAE: {t2m_metrics[f't+{t}h']['mae']:.4f}, "
                    f"RMSE: {t2m_metrics[f't+{t}h']['rmse']:.4f}, MAPE: {t2m_metrics[f't+{t}h']['mape']:.2f}%, "
                    f"RSE: {t2m_metrics[f't+{t}h']['rse']:.4f}, R²: {t2m_metrics[f't+{t}h']['r2']:.4f}\n")
            f.write(f"  RH2M - MSE: {rh2m_metrics[f't+{t}h']['mse']:.4f}, MAE: {rh2m_metrics[f't+{t}h']['mae']:.4f}, "
                    f"RMSE: {rh2m_metrics[f't+{t}h']['rmse']:.4f}, MAPE: {rh2m_metrics[f't+{t}h']['mape']:.2f}%, "
                    f"RSE: {rh2m_metrics[f't+{t}h']['rse']:.4f}, R²: {rh2m_metrics[f't+{t}h']['r2']:.4f}\n")

        f.write("\n[AVERAGES]\n")
        f.write(f"\n* T2M:\n  MSE: {avg_t2m['mse']:.4f}, MAE: {avg_t2m['mae']:.4f}, "
                f"RMSE: {avg_t2m['rmse']:.4f}, MAPE: {avg_t2m['mape']:.2f}%, "
                f"RSE: {avg_t2m['rse']:.4f}, R²: {avg_t2m['r2']:.4f}\n")
        f.write(f"\n* RH2M:\n  MSE: {avg_rh2m['mse']:.4f}, MAE: {avg_rh2m['mae']:.4f}, "
                f"RMSE: {avg_rh2m['rmse']:.4f}, MAPE: {avg_rh2m['mape']:.2f}%, "
                f"RSE: {avg_rh2m['rse']:.4f}, R²: {avg_rh2m['r2']:.4f}\n")
        f.write(f"\n* GENERAL:\n  MSE: {avg_general['mse']:.4f}, MAE: {avg_general['mae']:.4f}, "
                f"RMSE: {avg_general['rmse']:.4f}, MAPE: {avg_general['mape']:.2f}%, "
                f"RSE: {avg_general['rse']:.4f}, R²: {avg_general['r2']:.4f}\n")

    # Comparison plots
    plt.figure(figsize=(18, 12))
    for i, t in enumerate(targets):
        plt.subplot(2, 3, i + 1)
        plt.plot(y_true[:100, i], label='Actual', color='navy')
        plt.plot(y_pred[:100, i], '--', label=f'Pred (R²={t2m_metrics[f"t+{t}h"]["r2"]:.3f})', color='firebrick')
        plt.title(f'T2M - t+{t}h')
        plt.legend()

        plt.subplot(2, 3, i + 4)
        plt.plot(y_true[:100, i + 3], label='Actual', color='darkgreen')
        plt.plot(y_pred[:100, i + 3], '--', label=f'Pred (R²={rh2m_metrics[f"t+{t}h"]["r2"]:.3f})', color='orange')
        plt.title(f'RH2M - t+{t}h')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"comparison_{report_type}.png"), dpi=300)
    plt.close()

# ===================== LOSS CURVES PLOT =====================
def plot_loss_curves(history, save_dir):
    plt.figure(figsize=(12, 6))

    # Loss (MSE) plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training', color='blue')
    plt.plot(history.history['val_loss'], label='Validation', color='red')
    plt.title('MSE Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training', color='green')
    plt.plot(history.history['val_mae'], label='Validation', color='orange')
    plt.title('MAE Evolution')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=300)
    plt.close()

plot_loss_curves(history, val_plots_dir)

# ===================== EVALUATION =====================
best_model = keras.models.load_model(model_path)

# Validation evaluation
y_val_pred = best_model.predict(X_val, verbose=0)
generate_report(y_val, y_val_pred, targets, val_plots_dir, 'val')

# Test evaluation
y_test_pred = best_model.predict(X_test, verbose=0)
generate_report(y_test, y_test_pred, targets, test_plots_dir, 'test')

# ===================== TFLITE CONVERSION =====================
def convert_to_tflite():
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        converter.target_spec.supported_types = [tf.float32]
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()

        tflite_path = os.path.join(model_dir, "model.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"\nTFLite model saved at: {tflite_path}")

    except Exception as e:
        print(f"\nConversion error: {str(e)}")
        exit(1)

convert_to_tflite()

# ===================== RESULTS =====================
print("\n" + "=" * 50)
print("PROCESS COMPLETED")
print(f"Keras model: {model_path}")
print(f"TFLite model: {os.path.join(model_dir, 'model.tflite')}")
print(f"Validation reports: {val_plots_dir}")
print(f"Test reports: {test_plots_dir}")
print("=" * 50)