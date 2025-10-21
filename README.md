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

The final stage of the project was to build and evaluate machine learning models to predict loan default risk. This involved new feature engineering, a comparative analysis of preprocessing pipelines, and a systematic evaluation of oversampling techniques to handle our imbalanced dataset.

### Feature Engineering

To enhance the models' predictive power, several new features were created:

* **`credit_history_length`**: The length of the borrower's credit history.
* **`loan_to_income_ratio`**: The ratio of the loan amount to the borrower's annual income.
* **`instalment_to_income_ratio`**: The ratio of the loan's monthly installment to the borrower's estimated monthly income.

### Model Experiment Design

A comprehensive experiment was designed to determine the most effective combination of preprocessing, resampling, and modeling.

* **Models Tested**:
    * `LogisticRegression`
    * `RandomForestClassifier`
    * `LGBMClassifier` (LightGBM)

* **Resampling Techniques**:
    * `None` (original imbalanced data)
    * `SMOTE`
    * `ADASYN`
    * `SMOTE-Tomek`
    * `SMOTE-ENN`

* **Preprocessing Pipelines Compared**:
    * **Minimal Pipeline**: Included only basic imputation and one-hot encoding.
    * **Advanced Pipeline**: Included skewness correction (Box-Cox, Yeo-Johnson), feature scaling (StandardScaler), and PCA for dimensionality reduction.

### Results and Conclusion: Less is More

The results from the experiment revealed a clear and surprising trend: the **Minimal Preprocessing Pipeline consistently and significantly outperformed the Advanced Pipeline** across all models.

This suggests that for this dataset, aggressive transformations like skewness correction and PCA were detrimental, potentially distorting the original feature distributions. Tree-based models like LightGBM and Random Forest are inherently robust to skewed and unscaled data, which explains their strong performance in the minimal pipeline.

#### Key Takeaways:

* **🏆 Top Performer**: The best strategy was the **Minimal Pipeline + LightGBM + SMOTE**. This combination achieved an outstanding **F1-Score of 0.945** and a **Recall of 0.906** for the default class.
* **🚀 Highest Recall**: The highest recall (0.937) was achieved by the **Minimal Pipeline + LightGBM + SMOTE-ENN**, demonstrating the power of targeted oversampling.
* **💡 Simplicity Wins**: The minimal pipeline allowed models to learn directly from the data. The damage from over-processing was clear: Random Forest's recall, for example, dropped from **0.665** (Minimal) to a mere **0.022** (Advanced).

#### Performance Summary (Best F1-Score per Model)

The table below provides a "best-of-the-best" comparison. It shows the **single best-performing setup (based on highest F1-Score)** for each of the three models within each of the two pipelines. This clearly highlights the performance gap and supports the conclusion.

All metrics are for the minority class (default = 0).

| Pipeline | Model | Best Resampler | Precision (Default) | Recall (Default) | F1-Score (Default) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Minimal** | **LightGBM** | **SMOTE** | **0.988** | **0.906** | **0.945** |
| **Minimal** | **LogisticRegression** | **SMOTE-Tomek** | **0.980** | **0.904** | **0.940** |
| **Minimal** | **Random Forest** | **SMOTE-Tomek** | **0.937** | **0.665** | **0.778** |
| Advanced | LogisticRegression | SMOTE | 0.737 | 0.808 | 0.771 |
| Advanced | LightGBM | SMOTE-Tomek | 0.355 | 0.371 | 0.363 |
| Advanced | Random Forest | SMOTE-ENN | 0.218 | 0.717 | 0.335 |
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
