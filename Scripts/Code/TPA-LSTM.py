import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.layers import Input, Dense, LSTM, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import os

# ====================== CONFIGURATION ======================
# Model parameters
TIME_STEPS = 24
FEATURES = 2
PREDICTIONS = [1, 3, 6]
EMBED_DIM = 64

# Training parameters
EPOCHS = 120    ##This model takes longer to converge compared to the other models 
BATCH_SIZE = 256
LR_INITIAL = 0.0001
REDUCE_LR_FACTOR = 0.85
REDUCE_LR_PATIENCE = 10
EARLY_STOP_PATIENCE = 15

# Portable paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_VAL_DIR = os.path.join(MODELS_DIR, "ValPlots")
PLOTS_TEST_DIR = os.path.join(MODELS_DIR, "TestPlots")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_VAL_DIR, exist_ok=True)
os.makedirs(PLOTS_TEST_DIR, exist_ok=True)

# File paths
PATH_MODEL = os.path.join(MODELS_DIR, "best_model.keras")
PATH_TRAIN = os.path.join(DATA_DIR, "train.csv")
PATH_VAL = os.path.join(DATA_DIR, "val.csv")
PATH_TEST = os.path.join(DATA_DIR, "test.csv")

# ====================== HELPER FUNCTIONS ======================
def create_windows(data, time_steps, predictions):
    """Create input sequences and target values from time series data"""
    features = data[["T2M", "RH2M"]].values
    X, y = [], []
    max_offset = max(predictions)
    for i in range(len(features) - time_steps - max_offset):
        X.append(features[i:i + time_steps])
        y.append([features[i + time_steps + offset - 1] for offset in predictions])
    return np.array(X), np.array(y)


def positional_encoding(seq_len, embed_dim):
    """Generate positional encodings for transformer architecture"""
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
    pe = np.zeros((seq_len, embed_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return tf.constant(pe, dtype=tf.float32)


def generate_metrics_report(y_true, y_pred, horizons, save_path):
    """Generate and save a comprehensive metrics report"""
    report = ["COMPLETE METRICS REPORT", "==================================================\n"]
    metrics_all = {"T2M": [], "RH2M": []}

    for h_idx, horizon in enumerate(horizons):
        report.append(f"T+{horizon}H:")
        for feat_idx, feat in enumerate(["T2M", "RH2M"]):
            y_true_h = y_true[:, h_idx, feat_idx]
            y_pred_h = y_pred[h_idx][:, feat_idx]

            mse = mean_squared_error(y_true_h, y_pred_h)
            mae = mean_absolute_error(y_true_h, y_pred_h)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_true_h, y_pred_h) * 100
            r2 = r2_score(y_true_h, y_pred_h)
            rse = 1 - r2

            metrics_all[feat].append([mse, mae, rmse, mape, rse, r2])
            report.append(
                f"  {feat}: MSE={mse:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
                f"MAPE={mape:.2f}%, RSE={rse:.4f}, R²={r2:.4f}"
            )
        report.append("")

    # Aggregated metrics
    for feat in ["T2M", "RH2M"]:
        agg_metrics = np.mean(metrics_all[feat], axis=0)
        report.append(
            f"{feat}:\n  MSE={agg_metrics[0]:.4f}, MAE={agg_metrics[1]:.4f}, "
            f"RMSE={agg_metrics[2]:.4f}, MAPE={agg_metrics[3]:.2f}%, "
            f"RSE={agg_metrics[4]:.4f}, R²={agg_metrics[5]:.4f}"
        )

    with open(save_path, "w") as f:
        f.write("\n".join(report))


def plot_predictions(y_true, y_pred, horizons, title, save_folder):
    """Plot and save prediction vs actual comparisons"""
    plt.figure(figsize=(15, 10))
    for h_idx, horizon in enumerate(horizons):
        plt.subplot(3, 1, h_idx + 1)
        plt.plot(y_true[:, h_idx, 0], label="Actual T2M", alpha=0.7)
        plt.plot(y_pred[h_idx][:, 0], label="Predicted T2M", linestyle="--")
        plt.plot(y_true[:, h_idx, 1], label="Actual RH2M", alpha=0.7)
        plt.plot(y_pred[h_idx][:, 1], label="Predicted RH2M", linestyle="--")
        plt.title(f"{title} - T+{horizon}H")
        plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/pred_vs_real_{title.lower()}.png")
    plt.close()

# ====================== MODEL ARCHITECTURE ======================
def build_tpa_lstm():
    """Build the TPA-LSTM model architecture"""
    inputs = Input(shape=(TIME_STEPS, FEATURES))
    positional_embed = positional_encoding(TIME_STEPS, EMBED_DIM)
    x = Dense(EMBED_DIM)(inputs) + positional_embed
    attn_output = MultiHeadAttention(num_heads=4, key_dim=EMBED_DIM)(x, x)
    x = Concatenate()([x, attn_output])
    lstm_output = LSTM(64, return_sequences=False)(x)
    outputs = [Dense(FEATURES)(lstm_output) for _ in range(len(PREDICTIONS))]
    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR_INITIAL), loss="mse")
    return model

# ====================== TRAINING ======================
print("\nLoading and preparing data...")
train_data = pd.read_csv(PATH_TRAIN)
val_data = pd.read_csv(PATH_VAL)
X_train, y_train = create_windows(train_data, TIME_STEPS, PREDICTIONS)
X_val, y_val = create_windows(val_data, TIME_STEPS, PREDICTIONS)

callbacks = [
    ModelCheckpoint(PATH_MODEL, monitor="val_loss", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=REDUCE_LR_FACTOR, 
                     patience=REDUCE_LR_PATIENCE, verbose=1),
    EarlyStopping(monitor="val_loss", patience=EARLY_STOP_PATIENCE, 
                 restore_best_weights=True, verbose=1)
]

print("\nBuilding and training model...")
model = build_tpa_lstm()
history = model.fit(
    X_train, [y_train[:, i] for i in range(len(PREDICTIONS))],
    validation_data=(X_val, [y_val[:, i] for i in range(len(PREDICTIONS))]),
    epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
)

# ====================== VALIDATION AND TESTING ======================
# 1. Generate validation report
print("\nEvaluating on validation set...")
model.load_weights(PATH_MODEL)
val_preds = model.predict(X_val, verbose=0)
generate_metrics_report(y_val, val_preds, PREDICTIONS, 
                       f"{PLOTS_VAL_DIR}/metrics_report_val.txt")
plot_predictions(y_val, val_preds, PREDICTIONS, "Validation", PLOTS_VAL_DIR)

# 2. Final evaluation on test set
print("\nEvaluating on test set...")
test_data = pd.read_csv(PATH_TEST)
X_test, y_test = create_windows(test_data, TIME_STEPS, PREDICTIONS)
test_preds = model.predict(X_test, verbose=0)
generate_metrics_report(y_test, test_preds, PREDICTIONS, 
                       f"{PLOTS_TEST_DIR}/metrics_report_test.txt")
plot_predictions(y_test, test_preds, PREDICTIONS, "Test", PLOTS_TEST_DIR)

print("\nTraining and evaluation completed successfully!")
