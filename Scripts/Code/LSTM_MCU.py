import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Reshape
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# ===================== CONFIGURATION =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
val_plots_dir = os.path.join(model_dir, "ValPlots")
test_plots_dir = os.path.join(model_dir, "TestPlots")
model_path = os.path.join(model_dir, "best_model.keras")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(val_plots_dir, exist_ok=True)
os.makedirs(test_plots_dir, exist_ok=True)

window_size = 24
pred_steps = [1, 3, 6]

# ===================== FUNCTIONS =====================
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None)) * 100)

def calculate_rse(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def load_data(file_path):
    df = pd.read_csv(file_path)
    norm_data = df[['T2M', 'RH2M']].values  # Solo mantenemos las 2 características
    return norm_data, MinMaxScaler()

def generate_sequences(data):
    X, y = [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X.append(data[i - window_size:i].flatten())  # Ahora 2 características x ventana
        y.append(np.concatenate([data[i + step][:2] for step in pred_steps]))
    return np.array(X), np.array(y)

def generate_report(y_true, y_pred, plots_path, report_type='val'):
    t2m_metrics = {f't+{t}h': {} for t in pred_steps}
    rh2m_metrics = {f't+{t}h': {} for t in pred_steps}

    for i, t in enumerate(pred_steps):
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

    def calculate_averages(metric_dict):
        averages = {}
        for metric in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']:
            values = [metric_dict[f't+{t}h'][metric] for t in pred_steps]
            averages[metric] = np.mean(values)
        return averages

    avg_t2m = calculate_averages(t2m_metrics)
    avg_rh2m = calculate_averages(rh2m_metrics)
    avg_general = {metric: np.mean([avg_t2m[metric], avg_rh2m[metric]]) for metric in avg_t2m}

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
    input_layer = Input(shape=(window_size * 2,))
    x = Reshape((window_size, 2))(input_layer)  
    x = LSTM(36, activation='relu', unroll=True)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    output = Dense(6, activation='linear')(x)  
    return Model(inputs=input_layer, outputs=output)


# ===================== TRAINING =====================
def main():
    train_data, _ = load_data(os.path.join(data_dir, "train.csv"))
    val_data, _ = load_data(os.path.join(data_dir, "val.csv"))
    test_data, _ = load_data(os.path.join(data_dir, "test.csv"))

    X_train, y_train = generate_sequences(train_data)
    X_val, y_val = generate_sequences(val_data)
    X_test, y_test = generate_sequences(test_data)

    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=5, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=32,
        callbacks=callbacks
    )

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

    generate_report(y_val, model.predict(X_val), val_plots_dir, 'val')
    generate_report(y_test, model.predict(X_test), test_plots_dir, 'test')

    model.save(model_path)
    print("Training completed. Model and reports saved.")

    # TFLite Conversion 
    print("\nConverting to TFLite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Edge Impulse
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

    print(f"TFLite model saved to: {tflite_path}")

    # Verify with a classical interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    print("TFLite model verified and ready for deployment")


if __name__ == "__main__":
    main()
