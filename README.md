# Smart City Traffic Capstone

A comprehensive data analytics and machine learning project focused on interstate traffic volume analysis, feature engineering, and predictive modeling.

## 📁 Repository Structure

```text
├── README.md                                    # Project documentation (this file)
├── requirements.txt                             # Python package dependencies
├── pipeline.log                                 # Root-level pipeline execution logs
├── data/                                        # Root data folder
│   ├── ft_Metro_Interstate_Traffic_Volume.csv   # Feature-engineered traffic dataset
│   ├── Metro_Interstate_Traffic_Volume_original.csv # Raw original dataset
│   ├── Metro_Interstate_Traffic_Volume.csv      # Working traffic dataset
│   ├── ml_Metro_Interstate_Traffic_Volume.csv   # Machine learning prepared dataset
│   └── traffic_rules.csv                        # Domain-specific reference rules
├── figures/                                     # Root visualization outputs
│   ├── avg_traffic_by_hour.png                  # Traffic trend by hour chart
│   ├── traffic_holiday.png                      # Holiday traffic comparison plot
│   └── weekday_vs_weekend.png                   # Day-of-week impact visualization
├── Part 1 Data Analytics/                       # SQL, Power BI, and analytical reporting
│   ├── Add_DD_MM COL.sql                        # SQL script for date formatting
│   ├── Descriptive_Statistics_Correlation.sql    # Statistical analysis queries
│   ├── Insight Report.pdf                       # Final analytical summary report
│   ├── Insights.docx                            # Draft of analytical findings
│   ├── Metro_Interstate_Traffic_Volume_Correlation.xlsx # Correlation matrices
│   ├── Metro_Interstate_Traffic_Volume_powerBI.xlsx  # Cleaned source data for Power BI
│   ├── Metro_Interstate_Traffic_Volume_Probability.xlsx # Probability and distribution modeling
│   ├── Metro_Interstate_Traffic_Volume.sql      # Main analytical database scripts
│   ├── Metro_Interstate_Traffic_Volume1.xlsx    # Secondary worksheet logs
│   ├── Power BI Traffic Intelligence Dashboard.pbix  # Interactive analytical dashboard
│   ├── SQL-Based Traffic Analysis.sqbpro        # SQLite/DB Browser project file
│   ├── Temp_Holiday.sql                         # Temperature and holiday correlation scripts
│   └── Yearly_Traffic_Volume_Ascending.sql      # Long-term sorting trend queries
├── Part 2_Python/                               # Python automation and ETL pipelines
│   ├── artifacts/                               # Step-specific saved processing metrics
│   ├── data/                                    # Part 2 working data storage
│   ├── figures/                                 # Part 2 exploratory plots
│   ├── mlruns/                                  # MLflow tracking logs for local scripts
│   ├── App.py                                   # Application entryway script
│   ├── Feature Engineering .py                  # Data manipulation and feature extraction
│   ├── mlflow.db                                # Backend SQL database for MLflow tracking
│   ├── pipeline.log                             # Python pipeline-specific execution log
│   ├── pipeline.py                              # Core automated script orchestrator
│   └── Visualisation.py                         # Graph and chart rendering scripts
└── Part 3_machine Learning/                     # Core model training and tracking
    ├── artifacts/                               # Saved model binaries and weights
    ├── data/                                    # Model evaluation datasets
    ├── mlartifacts/                             # MLflow model registry artifacts
    ├── mlruns/                                  # Experiment tracking and logging metrics
    ├── Bias and Fairness.pdf                    # Final bias and algorithmic fairness report
    ├── Main.py                                  # Primary script for managing model pipelines
    ├── mlflow.db                                # MLflow experiment management backend
    ├── Part3_Task1_Sup_ML.ipynb                 # Supervised Learning models (Regression/Classification)
    ├── Part3_Task2_UnSup_ML.ipynb               # Unsupervised Learning models (Clustering/Anomalies)
    ├── Part3_Task3_DL_LTSM.ipynb                # Deep Learning Time-Series models (LSTM)
    ├── Part3_Task4_MLFlow.ipynb                 # MLflow workflow integration tracking
    ├── Part3_Task4_run_experiments.py           # Script to batch run model parameter tests
    ├── Part3_Task5_Recommendation system        # Recommendation pipeline modules
    ├── response_170606902883.json               # Raw API mock payload / model response logs
    └── train_simple.py                          # Baseline structural model trainer
```

## 🛠️ Architecture Breakdown

* **Data Analytics (`Part 1`)**: Focuses on parsing raw CSVs, applying deterministic rules via SQL scripts, establishing statistical correlations, and generating corporate-ready intelligence reporting via **Power BI**.
* **Python Pipeline (`Part 2`)**: Packages the processing steps into reproducible script structures (`pipeline.py`), managing feature transforms outside notebooks for automation robustness.
* **Machine Learning (`Part 3`)**: Implements advanced workflows ranging from classic ML to **Deep Learning (LSTM)**. Tracking is managed seamlessly across local SQLite database structures via **MLflow** runs.

## 🚀 Execution Quickstart

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Review Data Analysis:** Open the `.pbix` dashboard or step through the SQL files inside `Part 1 Data Analytics/`.
3. **Run Processing Pipeline:** 
   ```bash
   python "Part 2_Python/pipeline.py"
   ```
4. **Train ML Experiments:** 
   ```bash
   python "Part 3_machine Learning/Part3_Task4_run_experiments.py"
   ```
