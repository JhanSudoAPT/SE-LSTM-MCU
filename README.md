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
  Data/
    Raw/
    Processed/

  Figures/

  Models/
    ModelsComparison/
    ModelsOnMCUs/

  Results/
    Comparison I/
      MetricsTXT/
      Plots/
    Comparison II/
      Metrics/
      Plots/

  Scripts/
    Code/
      data/
    Implementation/
    ProcessData/
      Data/
      DataNormalized/
      AnalisisData/
      PLOTS/

