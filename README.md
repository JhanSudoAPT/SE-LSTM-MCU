This project uses recurrent neural networks (RNN), specifically LSTM, SE-LSTM, LMU, and TPA-LSTM, 
for prediction tasks. The project includes data preprocessing, model training, and the comparison 
of their results in terms of performance metrics and visualization. The development environment for
the code was PyCharm and Arduino IDE. The directory structure is as follows:

-------------------------------------------------------------------------------------------

Git/Data
Contains both raw and preprocessed datasets for training and testing the models.

Git/Models
Stores trained models in .keras format.
Inside the Git/Models/ModelsOnMCUs subfolder, you'll find versions optimized for microcontrollers
(MCUs), along with a descriptive usage file. For instance, the file best_model_SE_LSTM.tflite 
is ready for deployment on an MCU.

Git/Results (Comparison I and Comparison II)
These folders contain:

Metrics: Text files with evaluation results from the testing phase.

Plots: Graphs for visual comparison of model performance.
Since visual differences between models are subtle, the metrics are provided separately for better clarity.

Git/Scripts/Code (Code Testing)
Output directories are generated automatically during execution.
If you're comparing different models, make sure to rename the output folders to avoid overwriting existing results.

Libraries and Requirements
Dependencies are listed in library.txt and requirements.txt.
If you download the Git/Scripts/Code folder, you should be able to run the .py files without issues, as long as the data folder is present.

Git/Scripts/Code/Implementation
The folder contains instructions in the explanatory_text file for MCU usage.

Git/ProcessData(Data Processing) 
This folder also includes library.txt and requirements.txt, plus visualizations showing:

The raw data

How the data was cleaned and prepared for model training

-------------------------------------------------------------------------------------------

Directory Structure

Git/
  Data/               # Raw and processed data
    Raw/
      Dataset2020-2025.csv
    Processed/
      train.csv
      val.csv
      test.csv

  Figures/            # Graphs and visualizations
    Embeddings_PCA_2D.png
    Embeddings_PCA_3D.png
    ESP32.png
    Flowchart.png
    algorithm_flowchart.png

  Models/             # Trained models
    ModelsComparison/
      best_model_LMU.keras
      best_model_LSTM.keras
      best_model_SE_LSTM.keras
      best_model_TPA_LSTM.keras
    ModelsOnMCUs/
      best_model_LSTM_MCU.keras
      best_model_SE_LSTM.tflite
      best_model_SE_LSTM_ON_MCU.keras
      explanatory_text.txt

  Results/            # Test results and metrics
    Comparison I/
      MetricsTXT/
        metrics_LMU.txt
        metrics_LSTM.txt
        metrics_SE_LSTM.txt
        metrics_TPA_LSTM.txt
      Plots/
        comparison_test_LMU.png
        comparison_test_LSTM.png
        comparison_test_SE_LSTM.png
        comparison_test_TPA_LSTM.png
    Comparison II/
      Metrics/
        metrics_LSTM.txt
        metrics_SE_LSTM.txt
      Plots/
        comparison_test_LSTM.png
        comparison_test_SE_LSTM.png

  Scripts/            # Source code
    Code/
      LMU.py
      LSTM.py
      LSTM_MCU.py
      SE-LSTM.py
      SE-LSTM_MCU.py
      TPA-LSTM.py
      Emb_MCU.py
      requirements.txt
      library.txt
      data/
        train.csv
        val.csv
        test.csv

    Implementation/
      ei-prueba-arduino-1.0.1.zip
      explanatory_text.txt
      RapidTest.ino

    ProcessData/
      ProcessData.py
      requirements.txt
      library.txt
      Data/
        Dataset2020-2025.csv
        train.csv
        val.csv
        test.csv
      DataNormalized/
        Dataset2020-2025_Cleaned.csv
        Dataset2020-2025_Normalized.csv
      AnalisisData/
        Boxplot_RH2M.png
        Boxplot_T2M.png
        Histogram_RH2M.png
        Histogram_T2M.png
        OutliersDetected.csv
      PLOTS/
        normalized_distributions.png
        TimeSeries_RH2M.png
        TimeSeries_T2M.png
