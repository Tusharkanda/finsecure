# Bankruptcy Prediction – Gradio-Based User Interface

This repository provides a **machine learning–based bankruptcy prediction tool for Small and Medium Enterprises (SMEs)**. The project focuses on predicting bankruptcy status from financial ratios while ensuring **high accuracy, interpretability, and usability** through a **Gradio-based interactive interface**.

The proposed ensemble model outperforms traditional approaches and presents results in an accessible, explainable, and actionable manner.

---

## Overview

### Objective

Develop an **interpretable and robust bankruptcy prediction system** using financial ratios, complemented by a user-friendly interface that generates **visual explanations and automated reports**.

### Key Contributions

* Ensemble model combining **Logistic Regression, Random Forest, and XGBoost** for superior predictive performance.
* **SHapley Additive exPlanations (SHAP)** to identify key financial ratios influencing predictions.
* **Automated narrative report generation (NLG)** for human-readable insights and recommendations.
* **Gradio-based interface** enabling real-time predictions and interpretability.

---

## Project Workflow

### 1. Data Preparation

* Dataset consists of **427 companies** with **66 financial features**.
* Data preprocessing steps include:

  * Handling missing values
  * Outlier detection and treatment
  * Feature normalization
  * **SMOTE** for addressing class imbalance

### 2. Model Development

* Implemented an **ensemble (Voting Classifier)** consisting of:

  * Logistic Regression
  * Random Forest
  * XGBoost
* Model performance evaluated using:

  * Accuracy
  * ROC AUC
  * F1-score
* **Final performance**:

  * Accuracy: **96.63%**
  * ROC AUC: **99.6%**

### 3. User Interface & Interpretability

* Built using **Gradio** for seamless user interaction.
* Allows users to input financial ratios and receive **real-time predictions**.
* Integrated **SHAP visualizations** to highlight top contributing features influencing the bankruptcy decision.

### 4. Automated Report Generation

* Uses **Natural Language Generation (NLG)** to produce structured and readable reports.
* Reports include:

  * Predicted bankruptcy status
  * Influential financial ratios
  * Interpretive insights
  * Actionable financial recommendations

---

## Getting Started

### Prerequisites

* Python **3.12**
* Required Python libraries (install via `pip`):

  ```bash
  pip install -r requirements.txt
  ```

### Running the Project

#### Model Training

* Run the following notebook to preprocess data and train the ensemble model:

  ```bash
  ModelTraining.ipynb
  ```

#### Launching the Gradio Interface

* Start the Gradio application locally using:

  ```bash
  python GradioApp.py
  ```
* Open the generated local link in your browser to access the interface.

---

## Results

* The **Voting Classifier** demonstrates high accuracy and consistent performance across all bankruptcy classes.
* **SHAP-based explanations** enhance transparency and trust by revealing the most influential financial indicators.
* Automatically generated reports provide **concise summaries and practical financial strategies** for decision-making.

---

## Conclusion

This project delivers a **robust, interpretable, and user-friendly bankruptcy prediction system** for SMEs. By integrating advanced ensemble learning with explainable AI and automated reporting, the tool serves as a practical resource for:

* Financial analysts
* Business owners
* Investors and stakeholders

The combination of **high predictive performance, explainability, and accessibility** makes this solution well-suited for real-world financial risk assessment.

---

## License

This project is intended for academic and research purposes. Please ensure compliance with data usage and licensing policies before deployment.
