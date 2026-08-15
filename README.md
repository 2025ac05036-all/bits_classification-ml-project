# Machine Learning Classification Models with Streamlit Web Application

# a. Problem Statement

The objective of this project is to implement and compare multiple machine learning classification models on a common classification dataset. The models are evaluated using Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

The implemented classification models are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes
5. Random Forest Classifier

The models are trained and evaluated on the same dataset using a train-test split. The best-performing model is identified based on the evaluation metrics. The trained models are subsequently used in an interactive Streamlit web application.

---

# b. Dataset Description

# Dataset: Breast Cancer Wisconsin (Diagnostic)

The Breast Cancer Wisconsin (Diagnostic) dataset is a binary classification dataset containing measurements computed from digitized images of fine needle aspirate (FNA) of breast masses.

The dataset contains:

Number of instances: 569
Number of features: 30
Number of classes: 2
Feature type: Numerical
Classification type: Binary classification

The two target classes are:

Malignant
Benign

The dataset satisfies the assignment requirements of at least 500 instances and at least 12 features.

An 80:20 stratified train-test split was used for model evaluation. Standardization was applied within pipelines for Logistic Regression and KNN.



# c. GitHub Repository Link

GitHub Repository:

[https://github.com/2025ac05036-all/bits_classification-ml-project](https://github.com/2025ac05036-all/bits_classification-ml-project?utm_source=chatgpt.com)

The repository contains the files required for the project, including the Streamlit application, Python dependencies, test data, and trained machine learning models.

---

# d. Models Used

The following five classification models were implemented on the same dataset:

# 1. Logistic Regression

Logistic Regression is a linear classification algorithm that estimates the probability of a sample belonging to a particular class. Feature standardization was applied before training.

# 2. Decision Tree Classifier

Decision Tree is a non-linear classification algorithm that recursively divides the data based on feature values to make predictions.

# 3. K-Nearest Neighbor (KNN) Classifier

KNN classifies an observation based on the classes of its nearest neighboring observations. Feature standardization was applied before training.

# 4. Gaussian Naive Bayes

Gaussian Naive Bayes is a probabilistic classifier based on Bayes' theorem and assumes that the numerical features follow Gaussian distributions within each class.

# 5. Random Forest Classifier

Random Forest is an ensemble learning algorithm that combines predictions from multiple decision trees to improve generalization and reduce overfitting.

---

# Comparison of Model Performance

The following table presents the evaluation results obtained on the test dataset.

| ML Model Name            | Accuracy |    AUC | Precision | Recall | F1 Score |    MCC |
| ------------------------ | -------: | -----: | --------: | -----: | -------: | -----: |
| Logistic Regression      |   0.9825 | 0.9954 |    0.9861 | 0.9861 |   0.9861 | 0.9623 |
| K-Nearest Neighbors      |   0.9561 | 0.9788 |    0.9589 | 0.9722 |   0.9655 | 0.9054 |
| Random Forest (Ensemble) |   0.9474 | 0.9937 |    0.9583 | 0.9583 |   0.9583 | 0.8869 |
| Gaussian Naive Bayes     |   0.9386 | 0.9878 |    0.9452 | 0.9583 |   0.9517 | 0.8676 |
| Decision Tree            |   0.9123 | 0.9157 |    0.9559 | 0.9028 |   0.9286 | 0.8174 |

---

# Observations on Model Performance

# Logistic Regression

Logistic Regression achieved the best overall performance. It obtained the highest Accuracy (0.9825), AUC (0.9954), Precision (0.9861), Recall (0.9861), F1 Score (0.9861), and MCC (0.9623). This indicates that the model was highly effective at distinguishing between the two classes in this dataset.

# Decision Tree

The Decision Tree obtained an Accuracy of 0.9123 and an AUC of 0.9157, which were the lowest among the five models. Its Recall of 0.9028 was also the lowest. Although its Precision was relatively high at 0.9559, its overall performance was lower than the other models.

# K-Nearest Neighbors

KNN performed well, achieving an Accuracy of 0.9561 and an F1 Score of 0.9655. Its Recall of 0.9722 was higher than Random Forest, Gaussian Naive Bayes, and Decision Tree. However, it performed below Logistic Regression overall.

# Gaussian Naive Bayes

Gaussian Naive Bayes achieved an Accuracy of 0.9386, AUC of 0.9878, and F1 Score of 0.9517. Its AUC was relatively high, indicating strong class-ranking performance, although its Accuracy and MCC were lower than those of Logistic Regression and KNN.

# Random Forest

Random Forest achieved an Accuracy of 0.9474 and an F1 Score of 0.9583. It achieved a strong AUC of 0.9937, which was close to Logistic Regression. However, its Accuracy, F1 Score, and MCC were lower than Logistic Regression and KNN on this test set.

---

# Overall Winner

# Logistic Regression

Logistic Regression is the overall best-performing model for this dataset.

It achieved the highest value for all six evaluation metrics:

Accuracy: 0.9825
AUC: 0.9954
Precision: 0.9861
Recall: 0.9861
F1 Score: 0.9861
MCC: 0.9623

Therefore, Logistic Regression is selected as the overall winner based on the obtained test-set results.

# Project Structure

```text
bits_classification-ml-project/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── k_nearest_neighbors.pkl
    ├── gaussian_naive_bayes.pkl
    └── random_forest.pkl
```

The trained models and test data are used by the Streamlit application for interactive predictions and model demonstration.
