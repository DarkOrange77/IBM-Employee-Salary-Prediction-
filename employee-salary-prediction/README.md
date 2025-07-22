## 💼 Employee Salary Prediction using Machine Learning

### 🔍 Overview

This project predicts whether a person earns **more than \$50K or not** based on demographic and employment-related features from the **Adult Census Income dataset**.

It includes:

* Real-world data preprocessing
* Multiple machine learning models (KNN, Logistic Regression, Random Forest, SVM, Gradient Boosting, Neural Net)
* Model comparison using accuracy and classification reports
* An interactive **Streamlit frontend** for live predictions

---

### ⚙️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib / Seaborn** (optional)
* **Streamlit**
* **Joblib**

---

### 🧠 ML Models Used

| Model                        | Purpose                            |
| ---------------------------- | ---------------------------------- |
| `KNeighborsClassifier`       | Instance-based learning            |
| `LogisticRegression`         | Linear baseline                    |
| `RandomForestClassifier`     | Tree-based ensemble                |
| `SVM`                        | Margin-based classification        |
| `GradientBoostingClassifier` | Boosted ensemble                   |
| `MLPClassifier`              | Neural Network for deeper learning |

All models were evaluated using accuracy score, classification report, and confusion matrix.

---

### 🧼 Data Preprocessing

* Label encoding of categorical features (`gender`, `occupation`, etc.)
* Scaling with **MinMaxScaler**
* Capping outliers in `capital-loss`
* Stratified train-test split to preserve class balance

---

### 📊 Model Comparison

Each model was trained and tested using a unified pipeline to ensure fairness. Accuracy scores were plotted using `matplotlib`.

![Model Comparison Chart](yourimage.png) <!-- optional -->

---

### 🌐 Streamlit App

Users can:

* Enter their own features (age, gender, occupation, etc.)
* Choose an ML model (KNN, Logistic, MLP, etc.)
* See live predictions (>=50K or <=50K income)

To run:

```bash
streamlit run app.py
```

---

### 💾 How to Use

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/employee-salary-prediction.git
   cd employee-salary-prediction
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

---

### 🏁 Conclusion

> This project goes beyond basic classification by comparing **multiple ML models**, following **best practices in preprocessing**, and offering a **clean frontend** for user interaction. It simulates the full lifecycle of a real-world ML application — from raw data to deployable product.

---

### 📁 Folder Structure

```
├── app.py                  # Streamlit frontend
├── model_comparison.ipynb  # Jupyter notebook with all models + graphs
├── knn_model.pkl           # Saved model files
├── scaler.pkl              # Saved scaler
├── adult 3.csv             # Dataset
├── requirements.txt
└── README.md
```