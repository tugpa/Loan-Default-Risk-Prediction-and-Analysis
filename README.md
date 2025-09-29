# Loan Default Risk Prediction and Analysis

## Description

This project aims to analyze, visualize, and build a predictive model for loan default risk. By exploring a dataset of loan payments, this project identifies key factors influencing loan status and develops a machine learning model to predict the likelihood of a loan default. This README provides a comprehensive overview of the project, including the data used, the analysis performed, the models built, and the results obtained.

---

### Project Workflow

The project is structured into three main parts, each contained in a separate Jupyter notebook:

1.  **Exploratory Data Analysis (EDA)**: Initial data exploration, cleaning, and preparation. This phase focuses on understanding the data's structure, handling missing values, and preparing it for analysis and modeling.
2.  **Analysis and Visualization**: In-depth analysis and visualization of the data to identify potential risk factors and understand the overall portfolio. This includes examining both discrete and continuous variables to uncover trends related to loan status.
3.  **Predictive Modeling**: Building and evaluating machine learning models to predict loan default. This notebook covers feature engineering, model training, and performance evaluation to create an effective predictive tool.

---

## Data

The dataset used in this project is `loan_payments.csv`, which contains information about various loans and their payment statuses. The data undergoes several cleaning and transformation processes, resulting in different versions used throughout the analysis, including `loan_payments_post_null_imputation.csv` and `loan_payments_transformed.csv`.

---

## Exploratory Data Analysis (EDA)

The EDA process involved several key steps to prepare the data for analysis:

* **Data Formatting**: Standardizing column values, such as in the `verification_status` column, and converting date-related columns to a consistent datetime format.
* **Handling Missing Values**: Identifying and addressing null values in the dataset. Columns with a high percentage of missing data were dropped, and for those with fewer nulls, rows with missing values were removed.
* **Feature Transformation**: Applying transformations to skewed numerical data to improve model performance. Both Box-Cox and Yeo-Johnson transformations were considered and applied where effective.
* **Outlier Removal**: Identifying and removing outliers from the dataset using methods such as the IQR (Interquartile Range) to ensure they do not skew the analysis and modeling results.

---

## Analysis and Visualization

The analysis and visualization phase focused on understanding the factors that influence loan risk. Key insights were drawn from comparing different subsets of loans (all loans, fully paid, charged off/defaulted, and risky).

### Key Findings

* **Discrete Variables**: Analysis of discrete columns like `grade`, `term`, `employment_length`, `home_ownership`, and `purpose` revealed patterns in loan outcomes. For instance, debt consolidation loans, while being the majority, were also found to be slightly more likely to be charged off or defaulted.
* **Continuous Variables**: Continuous data such as `annual_inc`, `int_rate`, `loan_amount`, and `dti` were analyzed using histograms to compare distributions and means across different loan statuses. This helped in identifying trends that highlight variables impacting the risk of loss.

---

## Predictive Modeling

The final stage of the project was to build a predictive model for loan default risk. This involved:

* **Feature Engineering**: Creating new features like `credit_history_length`, `loan_to_income_ratio`, and `instalment_to_income_ratio` to provide more relevant information to the models.
* **Model Selection**: Several classification models were trained and evaluated, including **Logistic Regression**, **Random Forest**, and **LightGBM**.
* **Handling Class Imbalance**: The dataset exhibited a significant class imbalance, with fewer instances of defaulted loans. To address this, multiple techniques were employed:
    * **Class Weight Balancing**: The models were trained using balanced class weights, which penalizes misclassifications of the minority class more heavily.
    * **Resampling Techniques**: To create a more balanced dataset, several advanced resampling techniques were implemented:
        * **SMOTE (Synthetic Minority Over-sampling Technique)**: Oversamples the minority class by creating synthetic samples.
        * **ADASYN (Adaptive Synthetic Sampling)**: A variation of SMOTE that generates more synthetic data for minority class samples that are harder to learn.
        * **SMOTE-Tomek**: A hybrid method that combines oversampling (SMOTE) with undersampling (Tomek Links) to clean the space from overlapping instances.
        * **SMOTE-ENN**: Another hybrid method that combines SMOTE with Edited Nearest Neighbors (ENN) to remove noise from the majority class.

### Results

The models were evaluated based on accuracy, precision, recall, and F1-score, with a focus on the performance for the minority class (predicting defaults). The following table summarizes the performance of the **Logistic Regression** model under different data balancing scenarios:

| Model / Technique           | Accuracy | Precision (Default) | Recall (Default) | F1-Score (Default) |
| --------------------------- | -------- | ------------------- | ---------------- | ------------------ |
| Baseline (Imbalanced)       | 0.8846   | 0.90                | 0.37             | 0.52               |
| Balanced Class Weights      | 0.8842   | 0.63                | 0.80             | 0.70               |
| Resampling with SMOTE       | 0.9066   | 0.70                | 0.79             | 0.74               |
| Resampling with ADASYN      | 0.8864   | 0.61                | 0.87             | 0.72               |
| Resampling with SMOTE-Tomek | 0.9062   | 0.70                | 0.79             | 0.74               |
| Resampling with SMOTE-ENN   | 0.4974   | 0.21                | 0.71             | 0.32               |

As shown, the resampling techniques significantly improved the model's ability to identify defaults. **SMOTE** and **SMOTE-Tomek** provided the best balance of precision and recall, achieving the highest F1-scores and demonstrating a strong ability to predict loan defaults without compromising overall accuracy.

---

## How to Use

To run this project, you will need to have Python and the necessary libraries installed. You can run the Jupyter notebooks in the following order:

1.  `EDA.ipynb`
2.  `Analysis_Visualisation.ipynb`
3.  `Predictive_Modelling.ipynb`

---

## Technologies Used

* **Python**: The primary programming language used for this project.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For machine learning models, preprocessing, and evaluation.
* **LightGBM**: For the Light Gradient Boosting Machine model.
* **Imbalanced-learn**: For handling class imbalance with techniques like SMOTE.
* **Matplotlib & Seaborn**: For data visualization.
* **Plotly**: For interactive visualizations.
* **Missingno**: For visualizing missing data.
* **Statsmodels**: For statistical analysis.
