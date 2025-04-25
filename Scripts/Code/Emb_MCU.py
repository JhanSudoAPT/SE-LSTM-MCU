import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

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
embedding_dim = 4  # 4 seasons


# ===================== FUNCTIONS =====================
def load_data(file_path):
    """Loads and preprocesses data (already normalized)."""
    df = pd.read_csv(file_path)
    df['season'] = df['MO'].apply(lambda x: (x + 6) % 12 // 3)  # 0: Summer, 1: Autumn, 2: Winter, 3: Spring

    # Extract already normalized data
    data = df[['T2M', 'RH2M']].values

    return data, df['season'].values


def generate_sequences(data, seasons):
    """Generates sequences for the model."""
    X_cont, X_season, y = [], [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X_cont.append(data[i - window_size:i])  # (24, 2)
        X_season.append(seasons[i - window_size:i])  # (24,)
        y.append(np.hstack([data[i + step] for step in pred_steps]).flatten())  # (6,)
    return np.array(X_cont), np.array(X_season), np.array(y)


def calculate_mape(y_true, y_pred):
    """Calculates MAPE avoiding division by zero."""
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100


def generate_report(y_true, y_pred, plots_path, report_type='val'):
    """Generates metrics report and plots including overall averages."""
    metrics = {}

    # Lists to store all metrics for averages
    all_t2m_mse, all_t2m_mae, all_t2m_rmse, all_t2m_mape, all_t2m_r2 = [], [], [], [], []
    all_rh2m_mse, all_rh2m_mae, all_rh2m_rmse, all_rh2m_mape, all_rh2m_r2 = [], [], [], [], []

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
            'r2': r2_score(t2m_true, t2m_pred)
        }

        # Calculate metrics for RH2M
        rh2m_metrics = {
            'mse': mean_squared_error(rh2m_true, rh2m_pred),
            'mae': mean_absolute_error(rh2m_true, rh2m_pred),
            'rmse': np.sqrt(mean_squared_error(rh2m_true, rh2m_pred)),
            'mape': calculate_mape(rh2m_true, rh2m_pred),
            'r2': r2_score(rh2m_true, rh2m_pred)
        }

        # Save metrics by horizon
        metrics[f't+{t}h'] = {
            'T2M': t2m_metrics,
            'RH2M': rh2m_metrics
        }

        # Accumulate metrics for averages
        all_t2m_mse.append(t2m_metrics['mse'])
        all_t2m_mae.append(t2m_metrics['mae'])
        all_t2m_rmse.append(t2m_metrics['rmse'])
        all_t2m_mape.append(t2m_metrics['mape'])
        all_t2m_r2.append(t2m_metrics['r2'])

        all_rh2m_mse.append(rh2m_metrics['mse'])
        all_rh2m_mae.append(rh2m_metrics['mae'])
        all_rh2m_rmse.append(rh2m_metrics['rmse'])
        all_rh2m_mape.append(rh2m_metrics['mape'])
        all_rh2m_r2.append(rh2m_metrics['r2'])

    # Calculate overall averages
    avg_t2m = {
        'mse': np.mean(all_t2m_mse),
        'mae': np.mean(all_t2m_mae),
        'rmse': np.mean(all_t2m_rmse),
        'mape': np.mean(all_t2m_mape),
        'r2': np.mean(all_t2m_r2)
    }

    avg_rh2m = {
        'mse': np.mean(all_rh2m_mse),
        'mae': np.mean(all_rh2m_mae),
        'rmse': np.mean(all_rh2m_rmse),
        'mape': np.mean(all_rh2m_mape),
        'r2': np.mean(all_rh2m_r2)
    }

    # Calculate combined average (T2M and RH2M together)
    avg_combined = {
        'mse': np.mean(all_t2m_mse + all_rh2m_mse),
        'mae': np.mean(all_t2m_mae + all_rh2m_mae),
        'rmse': np.mean(all_t2m_rmse + all_rh2m_rmse),
        'mape': np.mean(all_t2m_mape + all_rh2m_mape),
        'r2': np.mean(all_t2m_r2 + all_rh2m_r2)
    }

    # Text report
    with open(os.path.join(plots_path, f"{report_type}_report.txt"), "w") as f:
        f.write("METRICS REPORT\n")
        f.write("=" * 50 + "\n")

        # Write metrics by horizon
        for horizon, values in metrics.items():
            f.write(f"{horizon}:\n")
            f.write(
                f"  T2M - MSE: {values['T2M']['mse']:.4f}, MAE: {values['T2M']['mae']:.4f}, RMSE: {values['T2M']['rmse']:.4f}, MAPE: {values['T2M']['mape']:.2f}%, R²: {values['T2M']['r2']:.4f}\n")
            f.write(
                f"  RH2M - MSE: {values['RH2M']['mse']:.4f}, MAE: {values['RH2M']['mae']:.4f}, RMSE: {values['RH2M']['rmse']:.4f}, MAPE: {values['RH2M']['mape']:.2f}%, R²: {values['RH2M']['r2']:.4f}\n\n")

        # Write overall averages
        f.write("=" * 50 + "\n")
        f.write("OVERALL AVERAGES:\n")
        f.write(
            f"  T2M (Average) - MSE: {avg_t2m['mse']:.4f}, MAE: {avg_t2m['mae']:.4f}, RMSE: {avg_t2m['rmse']:.4f}, MAPE: {avg_t2m['mape']:.2f}%, R²: {avg_t2m['r2']:.4f}\n")
        f.write(
            f"  RH2M (Average) - MSE: {avg_rh2m['mse']:.4f}, MAE: {avg_rh2m['mae']:.4f}, RMSE: {avg_rh2m['rmse']:.4f}, MAPE: {avg_rh2m['mape']:.2f}%, R²: {avg_rh2m['r2']:.4f}\n")
        f.write(
            f"  COMBINED (T2M+RH2M) - MSE: {avg_combined['mse']:.4f}, MAE: {avg_combined['mae']:.4f}, RMSE: {avg_combined['rmse']:.4f}, MAPE: {avg_combined['mape']:.2f}%, R²: {avg_combined['r2']:.4f}\n")

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


def visualize_embeddings(weights, plots_path):
    """Visualizes embeddings with 2D and 3D PCA."""
    seasons = ["Winter", "Spring", "Summer", "Autumn"]

    # PCA 2D
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(weights)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=range(4), cmap='viridis', s=100)
    for i, txt in enumerate(seasons):
        plt.annotate(txt, (emb_2d[i, 0], emb_2d[i, 1]), xytext=(10, 5), textcoords='offset points')
    plt.title("Season Embeddings (PCA 2D)")
    plt.colorbar(scatter, ticks=[0, 1, 2, 3], label='Season')
    plt.savefig(os.path.join(plots_path, "pca_2d.png"))
    plt.close()

    # PCA 3D
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(weights)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=range(4), cmap='viridis', s=100)
    for i, txt in enumerate(seasons):
        ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2], txt, fontsize=9)
    plt.title("Season Embeddings (PCA 3D)")
    fig.colorbar(scatter, ticks=[0, 1, 2, 3], pad=0.1)
    plt.savefig(os.path.join(plots_path, "pca_3d.png"))
    plt.close()


# ===================== MODEL =====================
def build_model():
    """Builds LSTM model with embeddings."""
    # Input for continuous data (T, H)
    input_cont = Input(shape=(window_size, 2))  # (24, 2)

    # Input for seasons + embeddings
    input_season = Input(shape=(window_size,))  # (24,)
    emb = Embedding(4, embedding_dim, input_length=window_size)(input_season)  # (24, 4)

    # Concatenate
    concatenated = Concatenate()([input_cont, emb])  # (24, 6)

    # Modified layers (only added Dense of 8)
    x = LSTM(64, activation='relu')(concatenated)  # 64 neurons
    x = Dense(32, activation='relu')(x)  # Added layer
    x = Dense(8, activation='relu')(x)  # Added layer
    output = Dense(len(pred_steps) * 2, activation='linear')(x)

    return Model(inputs=[input_cont, input_season], outputs=output)


# ===================== TRAINING =====================
def main():
    # Load data (already normalized)
    train_data, train_seasons = load_data(os.path.join(data_dir, "train.csv"))
    val_data, val_seasons = load_data(os.path.join(data_dir, "val.csv"))
    test_data, test_seasons = load_data(os.path.join(data_dir, "test.csv"))

    # Generate sequences
    X_cont_train, X_season_train, y_train = generate_sequences(train_data, train_seasons)
    X_cont_val, X_season_val, y_val = generate_sequences(val_data, val_seasons)
    X_cont_test, X_season_test, y_test = generate_sequences(test_data, test_seasons)

    # Build model
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Extract and show initial embedding vectors (random initialization)
    embedding_layer = model.layers[2]  # Embedding layer is the third in your model (index 2)
    embedding_weights = embedding_layer.get_weights()[0]  # Weights are a matrix of (4, embedding_dim)
    print("\nInitial embedding vectors (random):")
    print(embedding_weights)  # Matrix of 4 rows (seasons) x embedding_dim columns

    # Train
    history = model.fit(
        [X_cont_train, X_season_train],
        y_train,
        validation_data=([X_cont_val, X_season_val], y_val),
        epochs=5,
        batch_size=32
    )

    # Extract and show learned embedding vectors
    embedding_weights_trained = embedding_layer.get_weights()[0]
    seasons = ["Summer", "Autumn", "Winter", "Spring"]  # 0: Summer, 1: Autumn, 2: Winter, 3: Spring
    print("\nLearned embeddings by season:")
    for i, season in enumerate(seasons):
        print(f"{season}: {embedding_weights_trained[i]}")

    # ===== 1. Loss Plots (MSE and MAE) =====
    plt.figure(figsize=(12, 6))

    # MSE plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train MSE', color='blue')
    plt.plot(history.history['val_loss'], label='Validation MSE', color='red')
    plt.title('MSE during training')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # MAE plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE', color='green')
    plt.plot(history.history['val_mae'], label='Validation MAE', color='orange')
    plt.title('MAE during training')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(val_plots_path, "loss_metrics.png"), dpi=300)
    plt.close()

    # ===== 2. Rest of the code =====
    # Extract and visualize embeddings
    embedding_layer = model.layers[2]
    embedding_weights = embedding_layer.get_weights()[0]
    visualize_embeddings(embedding_weights, val_plots_path)

    # Generate reports
    y_pred_val = model.predict([X_cont_val, X_season_val])
    generate_report(y_val, y_pred_val, val_plots_path, 'val')

    y_pred_test = model.predict([X_cont_test, X_season_test])
    generate_report(y_test, y_pred_test, test_plots_path, 'test')

    # Save model
    model.save(model_path)
    print("Training completed. Model and reports saved.")


if __name__ == "__main__":
    main()