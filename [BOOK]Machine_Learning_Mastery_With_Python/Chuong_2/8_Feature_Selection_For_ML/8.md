# Chapter 8: Feature Selection for Machine Learning

The data features that you use to train your machine learning models have a huge influence on the performance you can achieve. Irrelevant or partially relevant features can negatively impact model performance. In this chapter, you will discover automatic feature selection techniques that you can use to prepare your machine learning data in Python with scikit-learn.

## 8.1 Feature Selection

**Feature Selection** is the process of automatically selecting those features in your data that contribute most to the prediction variable or output in which you are interested. Having irrelevant features in your data can decrease the accuracy of many models, especially linear algorithms like linear and logistic regression.

**Benefits of Feature Selection:**
- **Reduces Overfitting**: Less redundant data means less opportunity to make decisions based on noise.
- **Improves Accuracy**: Less misleading data means modeling accuracy improves.
- **Reduces Training Time**: Less data means that algorithms train faster.

## 8.2 Univariate Selection

**Univariate Selection** uses statistical tests to select those features that have the strongest relationship with the output variable. The scikit-learn library provides the `SelectKBest` class that can be used with a suite of different statistical tests to select a specific number of features.

### Example:
- Uses the **chi-squared (chi2)** statistical test to select 4 of the best features from the Pima Indians onset of diabetes dataset.

## 8.3 Recursive Feature Elimination (RFE)

**Recursive Feature Elimination (RFE)** works by recursively removing attributes and building a model on those attributes that remain. It uses the model accuracy to identify which attributes (and combinations of attributes) contribute the most to predicting the target attribute.

### Example:
- RFE uses a logistic regression model to select the top 3 features. The choice of algorithm doesn’t matter much, as long as it is skillful and consistent.

## 8.4 Principal Component Analysis (PCA)

**Principal Component Analysis (PCA)** is a data reduction technique that uses linear algebra to transform the dataset into a compressed form. PCA allows you to select a number of principal components in the transformed result.

### Example:
- PCA transforms the dataset into 3 principal components, which reduces dimensionality while retaining key information.

## 8.5 Feature Importance

**Feature Importance** is a technique used to estimate the importance of features using models like **Random Forest** and **Extra Trees**. These models can provide a score for each feature, indicating its importance in predicting the target attribute.

### Example:
- The **ExtraTreesClassifier** provides a feature importance score for each attribute. The larger the score, the more important the feature.

## 8.6 Summary

In this chapter, you learned about four different automatic feature selection techniques for preparing machine learning data in Python with scikit-learn:

- **Univariate Selection**
- **Recursive Feature Elimination (RFE)**
- **Principal Component Analysis (PCA)**
- **Feature Importance**

### 8.6.1 Next

Next, you will explore resampling methods that can be used to evaluate machine learning algorithms' performance on unseen data.
