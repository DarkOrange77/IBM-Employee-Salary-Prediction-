

##  Employee Salary Prediction using Machine Learning

This project predicts whether an individual earns more than \$50K per year using demographic and employment-related features. The task is framed as a **binary classification problem**, and the solution was developed as part of the **IBM SkillsBuild AI Internship** program. The dataset used was provided as part of the internship resources.

---

###  Features

* Predicts income class (`>50K` or `<=50K`) based on individual attributes
* Covers a complete ML pipeline: preprocessing, training, evaluation, and deployment
* Multiple algorithms were implemented and compared:

  * Logistic Regression
  * K-Nearest Neighbors (KNN)
  * Random Forest
  * Support Vector Machine (SVM)
  * Gradient Boosting
  * Neural Network (MLPClassifier)
* Final model selected: **Multilayer Perceptron (Neural Network)**
* Deployed via a **Streamlit** interface for user interaction

---

###  Tech Stack

* **Python 3.8+**
* **Pandas**, **NumPy**, **Scikit-learn**
* **Matplotlib**, **Seaborn** (EDA and visualization)
* **Streamlit** (for deployment)
* **Jupyter Notebook / VS Code** (development environment)

---

###  Model Insights

* **Neural Network (MLPClassifier)** provided the highest accuracy (\~85%) on the test set
* Addressed outliers and scaled features using **MinMaxScaler**
* Categorical variables were **label encoded**
* Stratified train-test split used for balanced class distribution

---

###  Deployment

The best model was integrated into a **Streamlit** app that allows users to input demographic/work features and receive real-time predictions on whether the individual is likely to earn more than \$50K per year.

---

###  Future Scope

* Extend to **regression** for estimating actual salaries
* Add **hyperparameter tuning** and **model explainability** (e.g., SHAP values)
* Incorporate **feedback loops** to retrain with real-world data
* Explore **bias mitigation techniques** based on gender or race


