import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tcn import TCN

# ===================== CONF =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
plots_val_dir = os.path.join(model_dir, "ValPlots")
plots_test_dir = os.path.join(model_dir, "TestPlots")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plots_val_dir, exist_ok=True)
os.makedirs(plots_test_dir, exist_ok=True)

# ===================== Data =====================
train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
val_df = pd.read_csv(os.path.join(data_dir, "val.csv"))
test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

features = ["T2M", "RH2M"]
train_data = train_df[features].values
val_data = val_df[features].values
test_data = test_df[features].values


# ===================== Sequence generation =====================
def create_sequences(data, seq_length, targets):
    X, y = [], []
    for i in range(len(data) - seq_length - max(targets)):
        X.append(data[i:i + seq_length])
        y.append([data[i + seq_length + t, 0] for t in targets] +
                 [data[i + seq_length + t, 1] for t in targets])
    return np.array(X), np.array(y)


seq_length = 24
targets = [1, 3, 6]
X_train, y_train = create_sequences(train_data, seq_length, targets)
X_val, y_val = create_sequences(val_data, seq_length, targets)
X_test, y_test = create_sequences(test_data, seq_length, targets)


# ===================== TCN =====================
def build_tcn_model(input_shape, output_size):
    model = Sequential([
        Input(shape=input_shape),
        TCN(
            nb_filters=32,
            kernel_size=3,
            nb_stacks=2,
            dilations=[1, 2, 4, 8],
            padding='causal',
            use_skip_connections=False,
            return_sequences=False
        ),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(output_size, activation='linear')
    ])
    model.compile(optimizer=Adam(0.001), loss='mse', metrics=['mae'])
    return model


model = build_tcn_model((seq_length, len(features)), len(targets) * 2)

# ===================== Train =====================
callbacks = [
    ModelCheckpoint(os.path.join(model_dir, "best_model.keras"),
                    monitor='val_loss', save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80, #This model converges in 40 to 70 epochs.
    batch_size=64,
    callbacks=callbacks,
    verbose=1
)


# ===================== Evaluation functions =====================
def generate_metrics_report(y_true, y_pred, horizon_names, file_path):
    report = []
    report.append("FULL METRICS REPORT")
    report.append("=" * 50)
    report.append("T2M (temperature) / RH2M (humidity)\n")

    # BY TIME HORIZON
    report.append("BY TIME HORIZON:\n")
    for i, horizon in enumerate(horizon_names):
        # T2M Metrics
        t2m_true = y_true[:, i]
        t2m_pred = y_pred[:, i]
        t2m_mse = mean_squared_error(t2m_true, t2m_pred)
        t2m_mae = mean_absolute_error(t2m_true, t2m_pred)
        t2m_rmse = np.sqrt(t2m_mse)
        t2m_mape = np.mean(np.abs((t2m_true - t2m_pred) / t2m_true)) * 100
        t2m_rse = np.sum((t2m_true - t2m_pred) ** 2) / np.sum((t2m_true - np.mean(t2m_true)) ** 2)
        t2m_r2 = r2_score(t2m_true, t2m_pred)

        # RH2M Metrics
        rh2m_true = y_true[:, i + 3]
        rh2m_pred = y_pred[:, i + 3]
        rh2m_mse = mean_squared_error(rh2m_true, rh2m_pred)
        rh2m_mae = mean_absolute_error(rh2m_true, rh2m_pred)
        rh2m_rmse = np.sqrt(rh2m_mse)
        rh2m_mape = np.mean(np.abs((rh2m_true - rh2m_pred) / rh2m_true)) * 100
        rh2m_rse = np.sum((rh2m_true - rh2m_pred) ** 2) / np.sum((rh2m_true - np.mean(rh2m_true)) ** 2)
        rh2m_r2 = r2_score(rh2m_true, rh2m_pred)

        report.append(f"T+{horizon}H:")
        report.append(
            f"  T2M: MSE={t2m_mse:.4f}, MAE={t2m_mae:.4f}, RMSE={t2m_rmse:.4f}, MAPE={t2m_mape:.2f}%, RSE={t2m_rse:.4f}, R²={t2m_r2:.4f}")
        report.append(
            f"  RH2M: MSE={rh2m_mse:.4f}, MAE={rh2m_mae:.4f}, RMSE={rh2m_rmse:.4f}, MAPE={rh2m_mape:.2f}%, RSE={rh2m_rse:.4f}, R²={rh2m_r2:.4f}\n")

    # AVERAGE METRICS
    report.append("=" * 50)
    report.append("Average of metrics:\n")

    # T2M Average
    t2m_all = y_true[:, :3].flatten()
    t2m_pred_all = y_pred[:, :3].flatten()
    t2m_metrics = (
        mean_squared_error(t2m_all, t2m_pred_all),
        mean_absolute_error(t2m_all, t2m_pred_all),
        np.sqrt(mean_squared_error(t2m_all, t2m_pred_all)),
        np.mean(np.abs((t2m_all - t2m_pred_all) / t2m_all)) * 100,
        np.sum((t2m_all - t2m_pred_all) ** 2) / np.sum((t2m_all - np.mean(t2m_all)) ** 2),
        r2_score(t2m_all, t2m_pred_all)
    )

    # RH2M Average
    rh2m_all = y_true[:, 3:].flatten()
    rh2m_pred_all = y_pred[:, 3:].flatten()
    rh2m_metrics = (
        mean_squared_error(rh2m_all, rh2m_pred_all),
        mean_absolute_error(rh2m_all, rh2m_pred_all),
        np.sqrt(mean_squared_error(rh2m_all, rh2m_pred_all)),
        np.mean(np.abs((rh2m_all - rh2m_pred_all) / rh2m_all)) * 100,
        np.sum((rh2m_all - rh2m_pred_all) ** 2) / np.sum((rh2m_all - np.mean(rh2m_all)) ** 2),
        r2_score(rh2m_all, rh2m_pred_all)
    )

    # All Metrics
    all_metrics = (
        mean_squared_error(y_true, y_pred),
        mean_absolute_error(y_true, y_pred),
        np.sqrt(mean_squared_error(y_true, y_pred)),
        np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2),
        r2_score(y_true, y_pred)
    )

    report.append("T2M:")
    report.append(
        f"  MSE={t2m_metrics[0]:.4f}, MAE={t2m_metrics[1]:.4f}, RMSE={t2m_metrics[2]:.4f}, MAPE={t2m_metrics[3]:.2f}%, RSE={t2m_metrics[4]:.4f}, R²={t2m_metrics[5]:.4f}\n")

    report.append("RH2M:")
    report.append(
        f"  MSE={rh2m_metrics[0]:.4f}, MAE={rh2m_metrics[1]:.4f}, RMSE={rh2m_metrics[2]:.4f}, MAPE={rh2m_metrics[3]:.2f}%, RSE={rh2m_metrics[4]:.4f}, R²={rh2m_metrics[5]:.4f}\n")

    report.append("ALL:")
    report.append(
        f"  MSE={all_metrics[0]:.4f}, MAE={all_metrics[1]:.4f}, RMSE={all_metrics[2]:.4f}, MAPE={all_metrics[3]:.2f}%, RSE={all_metrics[4]:.4f}, R²={all_metrics[5]:.4f}")

    with open(file_path, 'w') as f:
        f.write("\n".join(report))


def plot_training_history(history, plot_dir):
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_history.png'))
    plt.close()


def plot_predictions(y_true, y_pred, targets, plot_dir, prefix):
    plt.figure(figsize=(18, 12))

    for i, t in enumerate(targets):
        # T2M plot
        plt.subplot(3, 2, i + 1)
        plt.plot(y_true[:100, i], label='True', color='navy')
        plt.plot(y_pred[:100, i], '--', label='Predicted', color='firebrick')
        plt.title(f'T2M Prediction - t+{t}h')
        plt.ylabel('Normalized Value')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # RH2M plot
        plt.subplot(3, 2, i + 4)
        plt.plot(y_true[:100, i + 3], label='True', color='darkgreen')
        plt.plot(y_pred[:100, i + 3], '--', label='Predicted', color='orange')
        plt.title(f'RH2M Prediction - t+{t}h')
        plt.ylabel('Normalized Value')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{prefix}_predictions.png'))
    plt.close()


# ===================== EVALUATION =====================
model = keras.models.load_model(
    os.path.join(model_dir, "best_model.keras"),
    custom_objects={'TCN': TCN}
)

# Train
plot_training_history(history, model_dir)

# Val
y_val_pred = model.predict(X_val)
generate_metrics_report(y_val, y_val_pred, targets, os.path.join(plots_val_dir, "val_metrics.txt"))
plot_predictions(y_val, y_val_pred, targets, plots_val_dir, "val")

# Test
y_test_pred = model.predict(X_test)
generate_metrics_report(y_test, y_test_pred, targets, os.path.join(plots_test_dir, "test_metrics.txt"))
plot_predictions(y_test, y_test_pred, targets, plots_test_dir, "test")

# ===================== TFLITE =====================

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(os.path.join(model_dir, 'model.tflite'), 'wb') as f:
    f.write(tflite_model)

print("Completed")