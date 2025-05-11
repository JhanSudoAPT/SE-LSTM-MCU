import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from sklearn.preprocessing import MinMaxScaler
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

# Precalculated embeddings
EMBEDDINGS = {
    0: [-0.02646518, 0.15076376, 0.02600956, -0.11206985],  # Summer
    1: [0.10870589, 0.08809305, 0.04230594, 0.03112356],    # Autumn
    2: [-0.0566958, 0.11723963, 0.07567911, 0.07725443],    # Winter
    3: [0.00286462, 0.1879527, -0.07661887, 0.01724336]     # Spring
}

# ===================== FUNCTIONS =====================
def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None)) * 100)


def calculate_rse(y_true, y_pred):
    """Calculates Relative Squared Error (RSE)."""
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def load_data(file_path):
    """Loads data and adds embeddings."""
    df = pd.read_csv(file_path)

    # Verified normalization
    norm_data = df[['T2M', 'RH2M']].values

    # Get embeddings by month
    seasons = df['MO'].apply(lambda x: (x % 12) // 3).values
    embeddings = np.array([EMBEDDINGS[e] for e in seasons])

    # Concatenate data + embeddings (6 features)
    data = np.hstack([norm_data, embeddings])

    return data, MinMaxScaler()  


def generate_sequences(data):
    """Generates 24h sequences with embeddings."""
    X, y = [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X.append(data[i - window_size:i].flatten())  # (144,)

        # Output: [T+1, T+3, T+6, H+1, H+3, H+6]
        y.append(np.concatenate([data[i + step][:2] for step in pred_steps]))

    return np.array(X), np.array(y)


def generate_report(y_true, y_pred, plots_path, report_type='val'):
    """Generates complete metrics report."""
    t2m_metrics = {f't+{t}h': {} for t in pred_steps}
    rh2m_metrics = {f't+{t}h': {} for t in pred_steps}

    # Metrics by horizon
    for i, t in enumerate(pred_steps):
        # T2M
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

        # RH2M
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

    # Calculate averages
    def calculate_averages(metric_dict):
        averages = {}
        for metric in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']:
            values = [metric_dict[f't+{t}h'][metric] for t in pred_steps]
            averages[metric] = np.mean(values)
        return averages

    avg_t2m = calculate_averages(t2m_metrics)
    avg_rh2m = calculate_averages(rh2m_metrics)

    # General average
    avg_general = {}
    for metric in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']:
        avg_general[metric] = np.mean([avg_t2m[metric], avg_rh2m[metric]])

    # Report in TXT
    with open(os.path.join(plots_path, f"report_{report_type}.txt"), "w") as f:
        f.write("=" * 50 + "\n")
        f.write(f"METRICS REPORT ({report_type.upper()})\n")
        f.write("=" * 50 + "\n\n")

        f.write("[METRICS BY HORIZON]\n")
        for t in pred_steps:
            f.write(f"\n* t+{t}h:\n")
            f.write(
                f"  T2M  - MSE: {t2m_metrics[f't+{t}h']['mse']:.4f}, MAE: {t2m_metrics[f't+{t}h']['mae']:.4f}, RMSE: {t2m_metrics[f't+{t}h']['rmse']:.4f}, MAPE: {t2m_metrics[f't+{t}h']['mape']:.2f}%, RSE: {t2m_metrics[f't+{t}h']['rse']:.4f}, R²: {t2m_metrics[f't+{t}h']['r2']:.4f}\n")
            f.write(
                f"  RH2M - MSE: {rh2m_metrics[f't+{t}h']['mse']:.4f}, MAE: {rh2m_metrics[f't+{t}h']['mae']:.4f}, RMSE: {rh2m_metrics[f't+{t}h']['rmse']:.4f}, MAPE: {rh2m_metrics[f't+{t}h']['mape']:.2f}%, RSE: {rh2m_metrics[f't+{t}h']['rse']:.4f}, R²: {rh2m_metrics[f't+{t}h']['r2']:.4f}\n")

        f.write("\n[AVERAGES]\n")
        f.write(
            f"\n* T2M:\n  MSE: {avg_t2m['mse']:.4f}, MAE: {avg_t2m['mae']:.4f}, RMSE: {avg_t2m['rmse']:.4f}, MAPE: {avg_t2m['mape']:.2f}%, RSE: {avg_t2m['rse']:.4f}, R²: {avg_t2m['r2']:.4f}\n")
        f.write(
            f"\n* RH2M:\n  MSE: {avg_rh2m['mse']:.4f}, MAE: {avg_rh2m['mae']:.4f}, RMSE: {avg_rh2m['rmse']:.4f}, MAPE: {avg_rh2m['mape']:.2f}%, RSE: {avg_rh2m['rse']:.4f}, R²: {avg_rh2m['r2']:.4f}\n")
        f.write(
            f"\n* GENERAL:\n  MSE: {avg_general['mse']:.4f}, MAE: {avg_general['mae']:.4f}, RMSE: {avg_general['rmse']:.4f}, MAPE: {avg_general['mape']:.2f}%, RSE: {avg_general['rse']:.4f}, R²: {avg_general['r2']:.4f}\n")

    # Prediction plots
    plt.figure(figsize=(18, 12))
    for i, t in enumerate(pred_steps):
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
    plt.savefig(os.path.join(plots_path, f"comparison_{report_type}.png"), dpi=300)
    plt.close()


# ===================== MODEL =====================
def build_model():
    """LSTM 64-32-16 architecture for ESP32."""
    input_layer = Input(shape=(window_size * 6,))  # (144,)
    x = Reshape((window_size, 6))(input_layer)  # (24, 6)
    x = LSTM(64, activation='relu', unroll=True)(x)  # (64)
    x = Dense(32, activation='relu')(x)  # (32)
    x = Dense(8, activation='relu')(x)  # (8)
    output = Dense(6, activation='linear')(x)  # (6)

    return Model(inputs=input_layer, outputs=output)


# ===================== TRAINING =====================
def main():
    # Load data
    train_data, _ = load_data(os.path.join(data_dir, "train.csv"))
    val_data, _ = load_data(os.path.join(data_dir, "val.csv"))
    test_data, _ = load_data(os.path.join(data_dir, "test.csv"))

    # Generate sequences
    X_train, y_train = generate_sequences(train_data)
    X_val, y_val = generate_sequences(val_data)
    X_test, y_test = generate_sequences(test_data)

    # Model
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Patience
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,    ##This model takes a little longer to converge than the other model, a around 80-120; overfitting is visble after more than 150 epochs. 
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
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color='green')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
    plt.title('MAE Evolution')
    plt.xlabel('Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(val_plots_dir, "loss_metrics.png"), dpi=300)
    plt.close()

    # Generate reports
    generate_report(y_val, model.predict(X_val), val_plots_dir, 'val')
    generate_report(y_test, model.predict(X_test), test_plots_dir, 'test')

    # Save model
    model.save(model_path)
    print("Training completed. Model and reports saved.")

    # ===================== TFLITE CONVERSION =====================
    print("\nConverting to TFLite format...")

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optional optimizations (reduce size, improve latency)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert model
    tflite_model = converter.convert()

    # Save TFLite model
    tflite_path = os.path.join(model_dir, "model.tflite")
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to: {tflite_path}")

    # Verify TFLite model (optional)
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("TFLite model verified and ready for deployment")

if __name__ == "__main__":
    main()
