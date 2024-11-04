SmartRisk Loan Evaluator aims to assist investors on peer-to-peer lending platforms by predicting the likelihood of a borrower repaying their loan. The project processes borrower data, visualizes trends, and applies machine learning algorithms to classify loans as low-risk or high-risk, ultimately supporting investment decisions in the microfinance sector.

Features:
Data Preprocessing: Cleans and structures the dataset for analysis.
Feature Extraction: Identifies key features influencing loan risk.
Predictive Modeling: Utilizes multiple machine learning algorithms to predict borrower risk.
Imbalance Handling: Techniques like SMOTE and NearMiss are used to manage imbalanced data.
Ensemble Modeling: Combines predictions from multiple models for improved accuracy.
Visualization: Graphical representations of key data insights.

Technologies Used:
Python (Programming Language)
Pandas and NumPy (Data Manipulation)
Scikit-Learn (Machine Learning)
Matplotlib and Seaborn (Data Visualization)
Imbalanced-learn (Data Resampling)
Jupyter Notebook (Development Environment)

Project Structure:

SmartRisk Loan Evaluator/
│
├── data/                     # Dataset and processed data files
├── notebooks/                # Jupyter notebooks for analysis and modeling
├── src/                      # Source code for the project
│   ├── data_preprocessing.py # Data cleaning and preprocessing script
│   ├── feature_extraction.py # Feature selection and extraction module
│   ├── model_training.py     # Script to train and test ML models
│   ├── ensemble_model.py     # Ensemble modeling and voting classifier script
│   └── evaluation.py         # Model evaluation and metrics calculation
├── README.md                 # Project overview and instructions
└── requirements.txt          # Python package requirements
Dataset
The dataset used is from the Lending Club and contains borrower information for peer-to-peer loans. It includes 74 columns and over 800,000 records with fields like loan amount, interest rate, borrower annual income, and loan purpose. Key target classes include "Fully Paid" or "Charged Off" status, which indicate whether the borrower has fully repaid the loan or defaulted.

Installation
Clone the repository:
git clone https://github.com/MadhaviNandan/SmartRisk Loan Evaluator.git
cd SmartRisk Loan Evaluator

Usage:
Data Preprocessing: Clean and preprocess the dataset by running data_preprocessing.py in the src folder.
Feature Extraction: Run feature_extraction.py to select key features that impact loan risk.
Train Models: Execute model_training.py to train the machine learning models.
Evaluate Models: Run evaluation.py to view model performance metrics.
Ensemble Modeling (Optional): Use ensemble_model.py to combine multiple models for improved predictions.

Models and Algorithms:
The following machine learning models are implemented to classify loan risk:

Decision Tree
Logistic Regression
K-Nearest Neighbors (KNN)
Random Forest
Ensemble Model: A voting classifier that combines results from multiple models to increase accuracy.
Evaluation Metrics
To evaluate the performance of each model, the following metrics are used:

Accuracy: Overall success rate of the model.
Precision: Measures the accuracy of positive predictions.
Recall: Measures the coverage of actual positive instances.
F1 Score: Harmonic mean of Precision and Recall.
Confusion Matrix: Provides insights into true positives, false positives, true negatives, and false negatives.
Future Enhancements
Hyperparameter Tuning: Fine-tune model parameters to improve accuracy.
Additional Models: Test other advanced algorithms, like XGBoost and SVM.

