import pandas as pd
import os
import matplotlib.pyplot as plt

# ===================== CONFIGURATION =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Main directories
data_dir = os.path.join(BASE_DIR, "Data")
analysis_dir = os.path.join(BASE_DIR, "AnalisisData")
processed_dir = os.path.join(BASE_DIR, "DataNormalized")
plots_dir = os.path.join(BASE_DIR, "PLOTS")

# Create directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(analysis_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# File paths
input_path = os.path.join(data_dir, "Dataset2020-2025.csv")
cleaned_path = os.path.join(processed_dir, "Dataset2020-2025_Cleaned.csv")
normalized_path = os.path.join(processed_dir, "Dataset2020-2025_Normalized.csv")


# ===================== PROCESSING FUNCTIONS =====================
def load_and_clean_data():
    """Load and clean data while preserving temporal order"""
    print("Loading and cleaning data...")
    df = pd.read_csv(input_path, skiprows=10)

    # Clean data and sort temporally
    df = df[(df["T2M"] != -999) & (df["RH2M"] != -999)]
    df = df.sort_values(by=['YEAR', 'MO', 'DY', 'HR'])
    df.to_csv(cleaned_path, index=False)
    return df


def analyze_data(df):
    """Generate analysis plots and detect outliers"""
    print("Analyzing data...")
    columns = ["T2M", "RH2M"]

    # Outlier detection using IQR
    outliers_dict = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_dict[col] = outliers

    # Save combined outliers
    outliers_combined = pd.concat(outliers_dict.values())
    outliers_combined.to_csv(os.path.join(analysis_dir, "OutliersDetected.csv"), index=False)

    # Generate histograms and boxplots
    for col in columns:
        # Histogram
        plt.figure(figsize=(8, 5))
        plt.hist(df[col], bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col}")
        plt.grid()
        plt.savefig(os.path.join(analysis_dir, f"Histograma_{col}.png"))
        plt.close()

        # Boxplot
        plt.figure(figsize=(6, 5))
        plt.boxplot(df[col], vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue"))
        plt.ylabel(col)
        plt.title(f"Boxplot of {col}")
        plt.grid()
        plt.savefig(os.path.join(analysis_dir, f"Boxplot_{col}.png"))
        plt.close()


def normalize_data(df):
    """Normalize data between 0 and 1"""
    print("Normalizing data...")
    stats = {
        "T2M_max": df["T2M"].max(),
        "T2M_min": df["T2M"].min(),
        "RH2M_max": df["RH2M"].max(),
        "RH2M_min": df["RH2M"].min()
    }

    # Normalization
    df["T2M"] = (df["T2M"] - stats["T2M_min"]) / (stats["T2M_max"] - stats["T2M_min"])
    df["RH2M"] = (df["RH2M"] - stats["RH2M_min"]) / (stats["RH2M_max"] - stats["RH2M_min"])
    df.to_csv(normalized_path, index=False)
    return df


def split_data(df):
    """Split data into train/val/test respecting temporal order"""
    print("Temporal data splitting...")
    total = len(df)

    # Sequential splitting
    train_end = int(0.8 * total)
    val_end = train_end + int(0.15 * total)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    # Save splits
    train_df.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(data_dir, "test.csv"), index=False)
    return train_df, val_df, test_df


def generate_temporal_plots(df):
    """Generate temporal series plots"""
    print("Generating temporal plots...")
    for col in ["T2M", "RH2M"]:
        plt.figure(figsize=(8, 4))
        plt.plot(df[col], alpha=0.5)
        plt.title(f"Serie Temporal de {col} Normalizado")
        plt.savefig(os.path.join(plots_dir, f"SerieTemporal_{col}.png"))
        plt.close()


# ===================== MAIN EXECUTION =====================
if __name__ == "__main__":
    # 1. Load and clean
    df = load_and_clean_data()

    # 2. Generate analysis plots and outliers
    analyze_data(df)

    # 3. Normalize data
    normalized_df = normalize_data(df)

    # 4. Split data
    train_df, val_df, test_df = split_data(normalized_df)

    # 5. Generate temporal plots
    generate_temporal_plots(normalized_df)

    print("\nProcess completed successfully!")
    print(f"Output locations:")
    print(f"- Analysis plots: {analysis_dir}")
    print(f"- Temporal plots: {plots_dir}")
    print(f"- Cleaned data: {cleaned_path}")
    print(f"- Normalized data: {normalized_path}")
    print(f"- Data splits: {data_dir}")