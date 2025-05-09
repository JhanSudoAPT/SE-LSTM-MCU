import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import L2

# ==================================================
# Hyperparameter Configuration
# ==================================================
HP = {
    'seq_length': 24,  # Input sequence length
    'targets': [1, 3, 6],  # Prediction horizons (in hours)
    'units_lmu1': 96,  # Neurons in first LMU layer
    'theta_lmu1': 12,  # Theta for LMU1 (seq_length//2)
    'units_lmu2': 72,  # Neurons in second LMU layer
    'theta_lmu2': 6,  # Theta for LMU2 (seq_length//4)
    'dense_units': [48, 24],  # Units in dense layers
    'batch_size': 64,
    'epochs': 80, 
    'learning_rate': 0.0007,
    'weight_decay': 0.001,
    'l2_reg': 0.01  # L2 regularization
}

# ==================================================
# Portable Path Configuration
# ==================================================
# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Define all paths relative to the script location
data_dir = os.path.join(BASE_DIR, "data")
models_dir = os.path.join(BASE_DIR, "models")
plots_val_dir = os.path.join(models_dir, "ValPlots")
plots_test_dir = os.path.join(models_dir, "TestPlots")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_val_dir, exist_ok=True)
os.makedirs(plots_test_dir, exist_ok=True)

# ==================================================
# LMU Class
# ==================================================
class LMU(tf.keras.layers.Layer):
    def __init__(self, units, theta, d, return_sequences=False, **kwargs):
        super(LMU, self).__init__(**kwargs)
        self.units = units
        self._theta_val = float(theta)
        self.d = d
        self.return_sequences = return_sequences

    def build(self, input_shape):
        self._build_input_shape = input_shape
        self.A = self.add_weight(shape=(self.d, self.d), initializer='orthogonal', trainable=True, name='A_matrix')
        self.B = self.add_weight(shape=(input_shape[-1], self.d), initializer='glorot_uniform', trainable=True, name='B_matrix')
        Q = np.zeros((self.d, self.units))
        for n in range(self.d):
            for m in range(self.units):
                if n > m and (n + m) % 2 == 1:
                    Q[n, m] = (2 * n + 1) * (1.0 / np.sqrt(self.d))
        self.Q = tf.constant(Q, dtype=tf.float32, name='Q_matrix')
        self.feature_mixer = Dense(self.units, activation='tanh')
        self.feature_mixer.build((None, self.units))
        self.built = True

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        state = tf.zeros((batch_size, self.d), dtype=tf.float32)
        outputs = tf.TensorArray(tf.float32, size=time_steps)

        def body(t, state, outputs):
            input_slice = inputs[:, t, :]
            state_update = tf.matmul(state, self.A) + tf.matmul(input_slice, self.B)
            new_state = (1.0 - 1.0 / self._theta_val) * state + (1.0 / self._theta_val) * state_update
            output = tf.matmul(new_state, self.Q)
            output = self.feature_mixer(output)
            return t + 1, new_state, outputs.write(t, output)

        _, final_state, outputs = tf.while_loop(
            cond=lambda t, *_: t < time_steps,
            body=body,
            loop_vars=(0, state, outputs)
        )
        outputs = outputs.stack()
        outputs = tf.transpose(outputs, [1, 0, 2])
        return outputs if self.return_sequences else final_state

    def get_config(self):
        config = super(LMU, self).get_config()
        config.update({'units': self.units, 'theta': self._theta_val, 'd': self.d, 'return_sequences': self.return_sequences})
        return config

# ==================================================
# Helper Functions
# ==================================================
def create_sequences(data, seq_length, targets):
    """Creates training sequences."""
    X, y = [], []
    max_target = max(targets)
    for i in range(len(data) - seq_length - max_target):
        X.append(data[i: i + seq_length])
        y.append([data[i + seq_length + t, 0] for t in targets] +
                 [data[i + seq_length + t, 1] for t in targets])
    return np.array(X), np.array(y)


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates MAPE with protection against division by zero."""
    epsilon = 1e-7
    return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100


def relative_squared_error(y_true, y_pred):
    """Calculates RSE."""
    return np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def generate_metrics_report(y_true, y_pred, targets, filename):
    """Generates text for metrics comparison"""
    metrics = {}
    for i, t in enumerate(targets):
        metrics[f't+{t}h'] = {
            'T2M': {
                'mse': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_true[:, i], y_pred[:, i]),
                'rmse': np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])),
                'mape': mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]),
                'rse': relative_squared_error(y_true[:, i], y_pred[:, i]),
                'r2': r2_score(y_true[:, i], y_pred[:, i])
            },
            'RH2M': {
                'mse': mean_squared_error(y_true[:, i + 3], y_pred[:, i + 3]),
                'mae': mean_absolute_error(y_true[:, i + 3], y_pred[:, i + 3]),
                'rmse': np.sqrt(mean_squared_error(y_true[:, i + 3], y_pred[:, i + 3])),
                'mape': mean_absolute_percentage_error(y_true[:, i + 3], y_pred[:, i + 3]),
                'rse': relative_squared_error(y_true[:, i + 3], y_pred[:, i + 3]),
                'r2': r2_score(y_true[:, i + 3], y_pred[:, i + 3])
            }
        }

    metrics['total'] = {
        'T2M': {
            'mse': mean_squared_error(y_true[:, :3].flatten(), y_pred[:, :3].flatten()),
            'mae': mean_absolute_error(y_true[:, :3].flatten(), y_pred[:, :3].flatten()),
            'rmse': np.sqrt(mean_squared_error(y_true[:, :3].flatten(), y_pred[:, :3].flatten())),
            'mape': mean_absolute_percentage_error(y_true[:, :3].flatten(), y_pred[:, :3].flatten()),
            'rse': relative_squared_error(y_true[:, :3].flatten(), y_pred[:, :3].flatten()),
            'r2': r2_score(y_true[:, :3].flatten(), y_pred[:, :3].flatten())
        },
        'RH2M': {
            'mse': mean_squared_error(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten()),
            'mae': mean_absolute_error(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten()),
            'rmse': np.sqrt(mean_squared_error(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten())),
            'mape': mean_absolute_percentage_error(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten()),
            'rse': relative_squared_error(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten()),
            'r2': r2_score(y_true[:, 3:].flatten(), y_pred[:, 3:].flatten())
        },
        'all': {
            'mse': mean_squared_error(y_true.flatten(), y_pred.flatten()),
            'mae': mean_absolute_error(y_true.flatten(), y_pred.flatten()),
            'rmse': np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())),
            'mape': mean_absolute_percentage_error(y_true.flatten(), y_pred.flatten()),
            'rse': relative_squared_error(y_true.flatten(), y_pred.flatten()),
            'r2': r2_score(y_true.flatten(), y_pred.flatten())
        }
    }

    # Generate report text
    report = "COMPLETE METRICS REPORT\n"
    report += "=" * 50 + "\n\n"
    report += "BY TIME HORIZON:\n"

    for horizon in metrics:
        if horizon == 'total':
            continue
        report += f"\n{horizon.upper()}:\n"
        for var in ['T2M', 'RH2M']:
            m = metrics[horizon][var]
            report += (f"  {var}: MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
                       f"MAPE={m['mape']:.2f}%, RSE={m['rse']:.4f}, R²={m['r2']:.4f}\n")

    report += "\n" + "=" * 50 + "\n"
    report += "AGGREGATED METRICS:\n"

    for var in ['T2M', 'RH2M', 'all']:
        name = 'ALL' if var == 'all' else var
        m = metrics['total'][var]
        report += (f"\n{name}:\n  MSE={m['mse']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}, "
                   f"MAPE={m['mape']:.2f}%, RSE={m['rse']:.4f}, R²={m['r2']:.4f}\n")

    # Save to file
    with open(filename, "w") as f:
        f.write(report)

    return metrics

# ==================================================
# Data Loading and Preparation
# ==================================================
def load_data(file_name):
    """Loads already normalized data from CSV; all networks use normalized data"""
    df = pd.read_csv(os.path.join(data_dir, file_name))
    return df[["T2M", "RH2M"]].values


print("\nLoading data...")
train_data = load_data("train.csv")
val_data = load_data("val.csv")
test_data = load_data("test.csv")

print("\nCreating sequences...")
X_train, y_train = create_sequences(train_data, HP['seq_length'], HP['targets'])
X_val, y_val = create_sequences(val_data, HP['seq_length'], HP['targets'])
X_test, y_test = create_sequences(test_data, HP['seq_length'], HP['targets'])

print(f"\nDataset dimensions:")
print(f"Train: X={X_train.shape}, y={y_train.shape}")
print(f"Val:   X={X_val.shape}, y={y_val.shape}")
print(f"Test:  X={X_test.shape}, y={y_test.shape}")

# ==================================================
# Model Construction
# ==================================================
print("\nBuilding model...")
model = Sequential([
    Input(shape=(HP['seq_length'], 2)),
    LMU(units=HP['units_lmu1'], theta=HP['theta_lmu1'], d=HP['units_lmu1'], return_sequences=True),
    LMU(units=HP['units_lmu2'], theta=HP['theta_lmu2'], d=HP['units_lmu2'], return_sequences=False),
    Dense(HP['dense_units'][0], activation='relu', kernel_regularizer=L2(HP['l2_reg'])),
    Dense(HP['dense_units'][1], activation='gelu'),
    Dense(6, activation='tanh')
])

optimizer = Adam(learning_rate=HP['learning_rate'], weight_decay=HP['weight_decay'])
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', tf.keras.metrics.RootMeanSquaredError()])
model.summary()

# ==================================================
# Callbacks
# ==================================================
callbacks = [
    ModelCheckpoint(os.path.join(models_dir, "best_model.keras"), monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=40, min_delta=0.0001, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
]

# ==================================================
# Training
# ==================================================
print("\nStarting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val), epochs=HP['epochs'], batch_size=HP['batch_size'], callbacks=callbacks, verbose=1)

# ==================================================
# Training/Validation Plots
# ==================================================
print("\nSaving training plots...")
plt.figure(figsize=(12, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(plots_val_dir, "loss_history.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Training and Validation MAE')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend()
plt.savefig(os.path.join(plots_val_dir, "mae_history.png"), dpi=300)
plt.close()

# ==================================================
# Validation Set Evaluation
# ==================================================
print("\nEvaluating on validation set...")
y_val_pred = model.predict(X_val, verbose=1)
val_metrics = generate_metrics_report(y_val, y_val_pred, HP['targets'],
                                      os.path.join(plots_val_dir, "METRICS_REPORT_VAL.txt"))

# ==================================================
# Test Set Evaluation
# ==================================================
print("\nEvaluating on test set...")
y_test_pred = model.predict(X_test, verbose=1)
test_metrics = generate_metrics_report(y_test, y_test_pred, HP['targets'],
                                       os.path.join(plots_test_dir, "METRICS_REPORT_TEST.txt"))

# ==================================================
# Predictions vs Actual Plots (TEST)
# ==================================================
print("\nGenerating prediction plots...")
plt.figure(figsize=(18, 12))
for i, t in enumerate(HP['targets']):
    plt.subplot(3, 2, i + 1)
    plt.plot(y_test[:100, i], label='Actual', color='navy')
    plt.plot(y_test_pred[:100, i], '--', label=f'Pred (R²={test_metrics[f"t+{t}h"]["T2M"]["r2"]:.3f})', color='firebrick')
    plt.title(f'T2M - t+{t}h\nMSE: {test_metrics[f"t+{t}h"]["T2M"]["mse"]:.3f}, MAPE: {test_metrics[f"t+{t}h"]["T2M"]["mape"]:.2f}%')
    plt.legend()

for i, t in enumerate(HP['targets']):
    plt.subplot(3, 2, i + 4)
    plt.plot(y_test[:100, i + 3], label='Actual', color='darkgreen')
    plt.plot(y_test_pred[:100, i + 3], '--', label=f'Pred (R²={test_metrics[f"t+{t}h"]["RH2M"]["r2"]:.3f})', color='orange')
    plt.title(f'RH2M - t+{t}h\nMSE: {test_metrics[f"t+{t}h"]["RH2M"]["mse"]:.3f}, MAPE: {test_metrics[f"t+{t}h"]["RH2M"]["mape"]:.2f}%')
    plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(plots_test_dir, "predictions_vs_actuals.png"), dpi=300)
plt.close()

print("Training and evaluation completed")
