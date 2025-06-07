import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Concatenate
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

#In this code, you can see how each station evolves. 
# As mentioned in the article, they are grouped in sets of three,
# so you can change the assignment given in this code
# (which is configured for the Southern Hemisphere)."

# ===================== DIRECTORY CONFIGURATION =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(BASE_DIR, "data")
model_dir = os.path.join(BASE_DIR, "models")
val_plots_path = os.path.join(model_dir, "ValPlots")
test_plots_path = os.path.join(model_dir, "TestPlots")
embedding_plots_path = os.path.join(model_dir, "EmbeddingEvolution")
model_path = os.path.join(model_dir, "best_model.keras")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
os.makedirs(val_plots_path, exist_ok=True)
os.makedirs(test_plots_path, exist_ok=True)
os.makedirs(embedding_plots_path, exist_ok=True)

# ===================== HYPERPARAMETERS =====================
window_size = 24
pred_steps = [1, 3, 6]
embedding_dim = 4
SEASON_ORDER = ["Summer", "Autumn", "Winter", "Spring"]  # Southern Hemisphere


# ===================== DATA PREPROCESSING FUNCTIONS =====================
def load_data(file_path):
    """Loads and preprocesses data for Southern Hemisphere"""
    df = pd.read_csv(file_path)
    season_map = {
        1: 0, 2: 0, 12: 0,  # Summer: Dec, Jan, Feb
        3: 1, 4: 1, 5: 1,  # Autumn: Mar, Apr, May
        6: 2, 7: 2, 8: 2,  # Winter: Jun, Jul, Aug
        9: 3, 10: 3, 11: 3  # Spring: Sep, Oct, Nov
    }
    df['season'] = df['MO'].map(season_map)
    data = df[['T2M', 'RH2M']].values
    return data, df['season'].values


def generate_sequences(data, seasons):
    """Generates input sequences and target values"""
    X_cont, X_season, y = [], [], []
    for i in range(window_size, len(data) - max(pred_steps)):
        X_cont.append(data[i - window_size:i])
        X_season.append(seasons[i - window_size:i])
        y.append(np.hstack([data[i + step] for step in pred_steps]).flatten())
    return np.array(X_cont), np.array(X_season), np.array(y)


# ===================== METRIC FUNCTIONS =====================
def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-10, None))) * 100


def generate_report(y_true, y_pred, plots_path, report_type='val'):
    """Generates evaluation metrics and visualizations"""
    metrics = {}
    all_t2m_mse, all_t2m_mae, all_t2m_rmse, all_t2m_mape, all_t2m_r2 = [], [], [], [], []
    all_rh2m_mse, all_rh2m_mae, all_rh2m_rmse, all_rh2m_mape, all_rh2m_r2 = [], [], [], [], []

    for i, t in enumerate(pred_steps):
        # T2M metrics
        t2m_true = y_true[:, i]
        t2m_pred = y_pred[:, i]
        t2m_metrics = {
            'mse': mean_squared_error(t2m_true, t2m_pred),
            'mae': mean_absolute_error(t2m_true, t2m_pred),
            'rmse': np.sqrt(mean_squared_error(t2m_true, t2m_pred)),
            'mape': calculate_mape(t2m_true, t2m_pred),
            'r2': r2_score(t2m_true, t2m_pred)
        }

        # RH2M metrics
        rh2m_true = y_true[:, i + 3]
        rh2m_pred = y_pred[:, i + 3]
        rh2m_metrics = {
            'mse': mean_squared_error(rh2m_true, rh2m_pred),
            'mae': mean_absolute_error(rh2m_true, rh2m_pred),
            'rmse': np.sqrt(mean_squared_error(rh2m_true, rh2m_pred)),
            'mape': calculate_mape(rh2m_true, rh2m_pred),
            'r2': r2_score(rh2m_true, rh2m_pred)
        }

        metrics[f't+{t}h'] = {'T2M': t2m_metrics, 'RH2M': rh2m_metrics}

        # Plot predictions vs actual
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(t2m_true, t2m_pred, alpha=0.5)
        plt.plot([min(t2m_true), max(t2m_true)], [min(t2m_true), max(t2m_true)], 'r--')
        plt.xlabel('Actual Temperature')
        plt.ylabel('Predicted Temperature')
        plt.title(f'Temperature @ t+{t}h (R²={t2m_metrics["r2"]:.3f})')

        plt.subplot(1, 2, 2)
        plt.scatter(rh2m_true, rh2m_pred, alpha=0.5)
        plt.plot([min(rh2m_true), max(rh2m_true)], [min(rh2m_true), max(rh2m_true)], 'r--')
        plt.xlabel('Actual Humidity')
        plt.ylabel('Predicted Humidity')
        plt.title(f'Humidity @ t+{t}h (R²={rh2m_metrics["r2"]:.3f})')

        plt.tight_layout()
        plt.savefig(os.path.join(plots_path, f"{report_type}_prediction_{t}h.png"))
        plt.close()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Horizon': [f't+{t}h' for t in pred_steps],
        'T2M_MSE': [metrics[f't+{t}h']['T2M']['mse'] for t in pred_steps],
        'T2M_MAE': [metrics[f't+{t}h']['T2M']['mae'] for t in pred_steps],
        'T2M_RMSE': [metrics[f't+{t}h']['T2M']['rmse'] for t in pred_steps],
        'T2M_MAPE': [metrics[f't+{t}h']['T2M']['mape'] for t in pred_steps],
        'T2M_R2': [metrics[f't+{t}h']['T2M']['r2'] for t in pred_steps],
        'RH2M_MSE': [metrics[f't+{t}h']['RH2M']['mse'] for t in pred_steps],
        'RH2M_MAE': [metrics[f't+{t}h']['RH2M']['mae'] for t in pred_steps],
        'RH2M_RMSE': [metrics[f't+{t}h']['RH2M']['rmse'] for t in pred_steps],
        'RH2M_MAPE': [metrics[f't+{t}h']['RH2M']['mape'] for t in pred_steps],
        'RH2M_R2': [metrics[f't+{t}h']['RH2M']['r2'] for t in pred_steps]
    })
    metrics_df.to_csv(os.path.join(plots_path, f"{report_type}_metrics.csv"), index=False)


# ===================== EMBEDDING VISUALIZATION =====================
def visualize_embeddings(weights, plots_path):
    """Visualizes seasonal embeddings using PCA"""
    # 2D PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(weights)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=range(4), cmap='viridis', s=100)
    for i, txt in enumerate(SEASON_ORDER):
        plt.annotate(txt, (emb_2d[i, 0], emb_2d[i, 1]),
                     xytext=(10, 5), textcoords='offset points',
                     bbox=dict(facecolor='none', edgecolor='none', alpha=0.5))
    plt.title("Seasonal Embeddings (2D PCA)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    cbar = plt.colorbar(scatter, ticks=range(4))
    cbar.ax.set_yticklabels(SEASON_ORDER)
    plt.savefig(os.path.join(plots_path, "pca_2d.png"))
    plt.close()

    # 3D PCA
    pca = PCA(n_components=3)
    emb_3d = pca.fit_transform(weights)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2], c=range(4), cmap='viridis', s=100)
    for i, txt in enumerate(SEASON_ORDER):
        ax.text(emb_3d[i, 0], emb_3d[i, 1], emb_3d[i, 2], txt,
                fontsize=9, alpha=0.8,
                bbox=dict(facecolor='none', edgecolor='none', alpha=0.2))
    plt.title("Seasonal Embeddings (3D PCA)")
    fig.colorbar(scatter, ticks=range(4), pad=0.1)
    plt.savefig(os.path.join(plots_path, "pca_3d.png"))
    plt.close()


def plot_embedding_evolution(embedding_history, save_path):
    """Visualizes embedding evolution during training"""
    embedding_history = np.array(embedding_history)
    total_epochs = len(embedding_history) - 1

    all_projections = [PCA(n_components=2).fit_transform(epoch_emb) for epoch_emb in embedding_history]
    all_coords = np.concatenate(all_projections)
    x_min, x_max = all_coords[:, 0].min() - 0.1, all_coords[:, 0].max() + 0.1
    y_min, y_max = all_coords[:, 1].min() - 0.1, all_coords[:, 1].max() + 0.1

    plt.figure(figsize=(20, 15))
    key_epochs = [0, total_epochs // 4, total_epochs // 2, -1]
    epoch_labels = [0, total_epochs // 4, total_epochs // 2, total_epochs]

    for i, (epoch_idx, epoch_num) in enumerate(zip(key_epochs, epoch_labels)):
        plt.subplot(2, 2, i + 1)
        reduced_emb = PCA(n_components=2).fit_transform(embedding_history[epoch_idx])

        plt.scatter(reduced_emb[:, 0], reduced_emb[:, 1], c=range(4),
                    cmap='viridis', s=150, edgecolor='black')
        for j, txt in enumerate(SEASON_ORDER):
            plt.annotate(txt, (reduced_emb[j, 0], reduced_emb[j, 1]),
                         xytext=(5, 5), textcoords='offset points',
                         bbox=dict(facecolor='none', edgecolor='none', alpha=0.5))

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title(f'Epoch {epoch_num}')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    cbar_ax = plt.gcf().add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 3))
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks(range(4))
    cbar.set_ticklabels(SEASON_ORDER)
    cbar.set_label('Season', rotation=270, labelpad=15)

    plt.savefig(os.path.join(save_path, "embedding_evolution_combined.png"), dpi=150)
    plt.close()

    # Create animation
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    fig.subplots_adjust(right=0.85)

    # Configure colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 3))
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(4))
    cbar.ax.set_yticklabels(SEASON_ORDER)
    cbar.set_label('Season', rotation=270, labelpad=15)

    def update(frame):
        ax.clear()
        reduced_emb = PCA(n_components=2).fit_transform(embedding_history[frame])

        scatter = ax.scatter(reduced_emb[:, 0], reduced_emb[:, 1],
                             c=range(4), cmap='viridis',
                             s=250, edgecolor='black', linewidth=1.5, alpha=0.9)

        for j, txt in enumerate(SEASON_ORDER):
            ax.text(reduced_emb[j, 0], reduced_emb[j, 1] + (y_max - y_min) * 0.03, txt,
                    fontsize=10, ha='center', va='bottom',
                    bbox=dict(facecolor='none', edgecolor='none', alpha=0.5))

        ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max),
               title=f'Embedding Evolution - Epoch {frame}',
               xlabel='PC1', ylabel='PC2')
        ax.grid(alpha=0.2)
        return scatter,

    anim = FuncAnimation(fig, update, frames=len(embedding_history), interval=500)
    anim.save(os.path.join(save_path, "embedding_evolution.gif"),
              writer='pillow', fps=2, dpi=100)
    plt.close()


# ===================== MODEL ARCHITECTURE AND TRAINING =====================
def build_model():
    """Builds LSTM model with seasonal embeddings"""
    # Continuous inputs (temperature and humidity)
    input_cont = Input(shape=(window_size, 2))

    # Seasonal inputs (categorical)
    input_season = Input(shape=(window_size,))

    # Embedding layer for seasonal data
    emb = Embedding(4, embedding_dim, input_length=window_size)(input_season)

    # Concatenate continuous and embedded seasonal data
    concatenated = Concatenate()([input_cont, emb])

    # LSTM layers
    x = LSTM(36, activation='relu', return_sequences=False)(concatenated)
    x = Dense(128, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    # Output layer for multi-step predictions
    output = Dense(len(pred_steps) * 2, activation='linear')(x)

    return Model(inputs=[input_cont, input_season], outputs=output)


class EmbeddingHistory(Callback):
    """Tracks embedding evolution during training"""

    def __init__(self):
        super().__init__()
        self.history = []

    def on_train_begin(self, logs=None):
        """Capture initial weights before training"""
        embedding_layer = self.model.layers[2]
        self.history.append(embedding_layer.get_weights()[0].copy())

    def on_epoch_end(self, epoch, logs=None):
        """Capture weights after each epoch"""
        embedding_layer = self.model.layers[2]
        self.history.append(embedding_layer.get_weights()[0].copy())


def main():
    # Load and preprocess data
    print("Loading datasets...")
    train_data, train_seasons = load_data(os.path.join(data_dir, "train.csv"))
    val_data, val_seasons = load_data(os.path.join(data_dir, "val.csv"))
    test_data, test_seasons = load_data(os.path.join(data_dir, "test.csv"))

    # Generate training sequences
    print("Generating training sequences...")
    X_cont_train, X_season_train, y_train = generate_sequences(train_data, train_seasons)
    X_cont_val, X_season_val, y_val = generate_sequences(val_data, val_seasons)
    X_cont_test, X_season_test, y_test = generate_sequences(test_data, test_seasons)

    # Build and compile model
    print("Building model...")
    model = build_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Callbacks
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.85,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    embedding_callback = EmbeddingHistory()

    # Train model
    print("Starting model training...")
    history = model.fit(
        [X_cont_train, X_season_train],
        y_train,
        validation_data=([X_cont_val, X_season_val], y_val),
        epochs=80,
        batch_size=32,
        callbacks=[embedding_callback, lr_reducer]
    )

    # Generate visualizations and reports
    print("Generating embedding visualizations...")
    plot_embedding_evolution(embedding_callback.history, embedding_plots_path)
    visualize_embeddings(embedding_callback.history[-1], val_plots_path)

    print("Generating validation report...")
    y_pred_val = model.predict([X_cont_val, X_season_val])
    generate_report(y_val, y_pred_val, val_plots_path, 'val')

    print("Generating test report...")
    y_pred_test = model.predict([X_cont_test, X_season_test])
    generate_report(y_test, y_pred_test, test_plots_path, 'test')

    # Save model
    model.save(model_path)
    print("\nTraining completed successfully!")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()
