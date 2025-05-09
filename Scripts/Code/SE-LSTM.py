import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.layers import Dropout
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Concatenate
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Add
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# ===================== PORTABLE CONFIGURATION =====================
# Base directory (where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Relative paths (created automatically)
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
val_plots_path = os.path.join(model_dir, "ValPlots")
test_plots_path = os.path.join(model_dir, "TestPlots")
model_path = os.path.join(model_dir, "best_model.keras")

# Create directories if they don't exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(val_plots_path, exist_ok=True)
os.makedirs(test_plots_path, exist_ok=True)

# ===================== HYPERPARAMETERS =====================
window_size = 24
pred_steps = [1, 3, 6]
features = 6
embedding_dim = 4

# ===================== FUNCTIONS =====================
def load_data(file_path):
    df = pd.read_csv(file_path)
    seasons = df['MO'].apply(lambda x: ((x+6) % 12) // 3)   ###Encode months into seasonal groups
    df['season'] = seasons
    data = df[['T2M', 'RH2M']].values
    return data, seasons.values

def generate_sequences(data, seasons):
    X_cont, X_season, y = [], [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X_cont.append(data[i - window_size:i])
        X_season.append(seasons[i - window_size:i])
        y.append(np.hstack([data[i + step] for step in pred_steps]).flatten())
    return np.array(X_cont), np.array(X_season), np.array(y)

def visualize_embeddings(weights, plots_path):
    seasons = ["Autumn", "Winter", "Spring", "Summer"]
    pca_2d = PCA(n_components=2)
    emb_2d = pca_2d.fit_transform(weights)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=[0, 1, 2, 3], cmap='viridis', s=100)
    plt.colorbar(scatter, ticks=[0, 1, 2, 3], label='Season')
    plt.title("Season Embeddings (2D PCA)", fontweight='bold')
    plt.xlabel("Component 1", fontsize=10)
    plt.ylabel("Component 2", fontsize=10)

    for i, txt in enumerate(seasons):
        plt.annotate(txt, (emb_2d[i, 0], emb_2d[i, 1]), xytext=(10, 5), textcoords='offset points', fontsize=9)
    plt.grid(True)
    plt.savefig(os.path.join(plots_path, "Embeddings_PCA_2D.png"), dpi=300)
    plt.close()

    if weights.shape[1] >= 3:
        pca_3d = PCA(n_components=3)
        emb_3d = pca_3d.fit_transform(weights)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=[0, 1, 2, 3], cmap='viridis', s=100)
        plt.colorbar(scatter, ticks=[0, 1, 2, 3], pad=0.1, label='Season')
        ax.set_title("Season Embeddings (3D PCA)", fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=9)
        ax.set_ylabel("Component 2", fontsize=9)
        ax.set_zlabel("Component 3", fontsize=9)

        for i, txt in enumerate(seasons):
            ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2], txt, fontsize=8)
        plt.savefig(os.path.join(plots_path, "Embeddings_PCA_3D.png"), dpi=300)
        plt.close()

def generate_report(y_true, y_pred, plots_path, report_type='val'):
    metrics = {}
    for i, t in enumerate(pred_steps):
        temp_true = y_true[:, i]
        temp_pred = y_pred[:, i]
        hum_true = y_true[:, i + 3]
        hum_pred = y_pred[:, i + 3]

        metrics[f't+{t}h'] = {
            'Temperature': {
                'mse': mean_squared_error(temp_true, temp_pred),
                'mae': mean_absolute_error(temp_true, temp_pred),
                'rmse': np.sqrt(mean_squared_error(temp_true, temp_pred)),
                'mape': np.mean(np.abs((temp_true - temp_pred) / np.clip(np.abs(temp_true), 1e-10, None))) * 100,
                'rse': np.sum((temp_true - temp_pred)**2) / np.sum((temp_true - np.mean(temp_true))**2),
                'r2': r2_score(temp_true, temp_pred)
            },
            'Humidity': {
                'mse': mean_squared_error(hum_true, hum_pred),
                'mae': mean_absolute_error(hum_true, hum_pred),
                'rmse': np.sqrt(mean_squared_error(hum_true, hum_pred)),
                'mape': np.mean(np.abs((hum_true - hum_pred) / np.clip(np.abs(hum_true), 1e-10, None))) * 100,
                'rse': np.sum((hum_true - hum_pred)**2) / np.sum((hum_true - np.mean(hum_true))**2),
                'r2': r2_score(hum_true, hum_pred)
            }
        }

    report_path = os.path.join(plots_path, f"{report_type}_metrics.txt")
    with open(report_path, "w") as f:
        f.write("COMPLETE METRICS REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write("BY TIME HORIZON:\n\n")

        for horizon, values in metrics.items():
            f.write(f"{horizon.upper()}:\n")
            temp = values['Temperature']
            hum = values['Humidity']
            f.write(f"  Temperature: MSE={temp['mse']:.4f}, MAE={temp['mae']:.4f}, RMSE={temp['rmse']:.4f}, MAPE={temp['mape']:.2f}%, RSE={temp['rse']:.4f}, R²={temp['r2']:.4f}\n")
            f.write(f"  Humidity: MSE={hum['mse']:.4f}, MAE={hum['mae']:.4f}, RMSE={hum['rmse']:.4f}, MAPE={hum['mape']:.2f}%, RSE={hum['rse']:.4f}, R²={hum['r2']:.4f}\n\n")

        f.write("=" * 50 + "\n")
        f.write("AGGREGATED METRICS:\n\n")

        temp_avg = {k: np.mean([v['Temperature'][k] for v in metrics.values()]) for k in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']}
        f.write("Temperature:\n")
        f.write(f"  MSE={temp_avg['mse']:.4f}, MAE={temp_avg['mae']:.4f}, RMSE={temp_avg['rmse']:.4f}, MAPE={temp_avg['mape']:.2f}%, RSE={temp_avg['rse']:.4f}, R²={temp_avg['r2']:.4f}\n\n")

        hum_avg = {k: np.mean([v['Humidity'][k] for v in metrics.values()]) for k in ['mse', 'mae', 'rmse', 'mape', 'rse', 'r2']}
        f.write("Humidity:\n")
        f.write(f"  MSE={hum_avg['mse']:.4f}, MAE={hum_avg['mae']:.4f}, RMSE={hum_avg['rmse']:.4f}, MAPE={hum_avg['mape']:.2f}%, RSE={hum_avg['rse']:.4f}, R²={hum_avg['r2']:.4f}\n\n")

        global_avg = {k: (temp_avg[k] + hum_avg[k]) / 2 for k in temp_avg}
        f.write("ALL:\n")
        f.write(f"  MSE={global_avg['mse']:.4f}, MAE={global_avg['mae']:.4f}, RMSE={global_avg['rmse']:.4f}, MAPE={global_avg['mape']:.2f}%, RSE={global_avg['rse']:.4f}, R²={global_avg['r2']:.4f}\n")

    plt.figure(figsize=(18, 12))
    for i, t in enumerate(pred_steps):
        plt.subplot(3, 2, i + 1)
        plt.plot(y_true[:100, i], label='True', color='navy')
        plt.plot(y_pred[:100, i], '--', label=f'Pred (R²={metrics[f"t+{t}h"]["Temperature"]["r2"]:.3f})', color='firebrick')
        plt.title(f'Temperature - t+{t}h')
        plt.xlabel('Number of Samples')
        plt.ylabel('Normalized Temperature')
        plt.legend()

        plt.subplot(3, 2, i + 4)
        plt.plot(y_true[:100, i + 3], label='True', color='darkgreen')
        plt.plot(y_pred[:100, i + 3], '--', label=f'Pred (R²={metrics[f"t+{t}h"]["Humidity"]["r2"]:.3f})', color='orange')
        plt.title(f'Humidity - t+{t}h')
        plt.xlabel('Number of Samples')
        plt.ylabel('Normalized Humidity')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"Comparison_{report_type}.png"), dpi=300)
    plt.close()

def plot_training_history(history, plots_path):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', color='red', linewidth=2)
    plt.title('MSE Evolution', fontweight='bold')
    plt.ylabel('MSE')
    plt.xlabel('Epoch', fontsize=10)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training', color='blue', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation', color='red', linewidth=2)
    plt.title('MAE Evolution', fontweight='bold')
    plt.ylabel('MAE')
    plt.xlabel('Epoch', fontsize=10)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, "Training_Evolution.png"), dpi=300)
    plt.close()

# ===================== TRAINING =====================
train_data, train_seasons = load_data(os.path.join(data_dir, "train.csv"))
val_data, val_seasons = load_data(os.path.join(data_dir, "val.csv"))

X_cont_train, X_season_train, y_train = generate_sequences(train_data, train_seasons)
X_cont_val, X_season_val, y_val = generate_sequences(val_data, val_seasons)

def build_model():
    input_cont = Input(shape=(window_size, 2))
    input_season = Input(shape=(window_size,))
    emb = Embedding(4, embedding_dim, input_length=window_size)(input_season)
    concatenated = Concatenate()([input_cont, emb])
    x = LSTM(64, return_sequences=True, activation='relu')(concatenated)
    x = LSTM(32, return_sequences=True, activation='relu')(x)
    x = LSTM(8, activation='relu')(x)
    output = Dense(features, activation='linear')(x)
    model = Model(inputs=[input_cont, input_season], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

model = build_model()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.85, patience=10, min_delta=0.001, min_lr=1e-6, verbose=1)
history = model.fit([X_cont_train, X_season_train], y_train, epochs=90, batch_size=64,
                   validation_data=([X_cont_val, X_season_val], y_val), callbacks=[reduce_lr])

plot_training_history(history, val_plots_path)
model.save(model_path)

# ===================== VALIDATION =====================
y_pred_val = model.predict([X_cont_val, X_season_val])
generate_report(y_val, y_pred_val, val_plots_path, 'val')

embedding_layer = model.layers[2]
weights = embedding_layer.get_weights()[0]
visualize_embeddings(weights, val_plots_path)

# ===================== TEST =====================
test_data, test_seasons = load_data(os.path.join(data_dir, "test.csv"))
X_cont_test, X_season_test, y_test = generate_sequences(test_data, test_seasons)
y_pred_test = model.predict([X_cont_test, X_season_test])
generate_report(y_test, y_pred_test, test_plots_path, 'test')

print("Training and evaluation completed")
