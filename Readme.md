# Insider Trading Anomaly Detection

## Overview  
This project identifies insider trading anomalies in market transactions to support regulatory compliance. Using machine learning techniques, it analyzes corporate disclosures and high-frequency trade data to detect suspicious trading patterns that may indicate potential misuse of non-public information.

## Features  
- Data preprocessing from SEC EDGAR filings and trade logs  
- Detection of abnormal trading behaviors  
- Classification of trades as normal or high-risk  
- Generation of analytical reports and visualization graphs  
- Model evaluation using multiple performance metrics  

## Tech Stack  
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib  
- **Techniques:** Machine Learning, Classification, Anomaly Detection, Data Visualization  

## Dataset  
**Source:** SEC EDGAR Filings and Trade Data (Kaggle)  
This dataset includes corporate disclosure records and high-frequency trading logs used for identifying unusual market behavior.

## Model Workflow  
1. **Data Preprocessing:** Cleaning, feature selection, and normalization  
2. **Model Training:** Logistic Regression, Random Forest, and XGBoost classifiers  
3. **Evaluation:** Ensemble Voting Classifier with calibrated probabilities  
4. **Result Visualization:** ROC curve, Precision-Recall curve, and confusion matrix  

## Results  
The model demonstrated strong performance in distinguishing between normal and high-risk trades.  

**Performance Summary:**  
- **Accuracy:** 95.85%  
- **ROC-AUC:** 0.9824  
- **Precision:** 93.12%  
- **Recall (High-Risk Detection):** 92.55%  
- **F1-Score:** 92.83%  

These results indicate that the model effectively identifies potentially suspicious insider trading activities with high reliability and minimal false positives.

## Output  
- Predicted classification of trades as **Normal** or **High-Risk**  
- Comprehensive model performance metrics  
- Visual insights through ROC and Precision-Recall curves  
- Support for compliance teams to investigate flagged trades  

## Author  
**Mohit Mehta**
