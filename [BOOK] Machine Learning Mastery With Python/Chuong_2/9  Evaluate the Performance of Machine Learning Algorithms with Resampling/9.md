# Chapter 9 - Evaluate the Performance of Machine Learning Algorithms with Resampling

In this chapter, you will discover how to estimate the accuracy of machine learning algorithms using resampling methods in Python and scikit-learn. The goal is to evaluate how well your algorithms perform on unseen data by using statistical techniques called **resampling**. Here are the main techniques discussed in this chapter:

## 9.1 Evaluate Machine Learning Algorithms

When evaluating machine learning algorithms, it’s important to avoid overfitting. Overfitting occurs when an algorithm memorizes the data it is trained on, leading to high performance on training data but poor performance on new, unseen data. To avoid overfitting, we need to evaluate the algorithm on data that was not part of the training process. This evaluation gives us an estimate of how the model might perform on new data.

There are several resampling techniques to estimate the performance of a machine learning model:

- **Train and Test Sets**
- **K-fold Cross Validation**
- **Leave One Out Cross Validation**
- **Repeated Random Test-Train Splits**

## 9.2 Split into Train and Test Sets

This technique splits the dataset into two parts: one for training and one for testing. The model is trained on the training set and tested on the test set. A typical split is **67% training** and **33% testing**. While this method is fast and ideal for large datasets, it has high variance. This means the results can change depending on the specific data split.

### Advantages:
- Fast and simple.
- Ideal for large datasets.

### Disadvantages:
- High variance in performance estimates due to the random split.

## 9.3 K-fold Cross Validation

In K-fold cross-validation, the dataset is split into **k** parts (folds). The model is trained on **k-1** folds and tested on the remaining fold. This process is repeated for each fold, giving us **k** performance scores. The mean and standard deviation of these scores provide a more reliable estimate of model performance.

### Advantages:
- More reliable than a single train-test split.
- Reduces variance by using different subsets for training and testing.

### Disadvantages:
- Computationally expensive, especially for large datasets.

## 9.4 Leave One Out Cross Validation (LOO-CV)

Leave-One-Out Cross Validation (LOO-CV) is a special case of K-fold cross-validation where **k** is equal to the number of observations in the dataset. Each instance is used as a test set once, while the rest of the data is used for training. While very thorough, this method is computationally expensive.

### Advantages:
- Very detailed estimate of model performance.
- Every data point is used for testing.

### Disadvantages:
- Computationally expensive.

## 9.5 Repeated Random Test-Train Splits

This technique involves repeating the random split of the dataset into training and test sets multiple times. This method combines the speed of the train-test split with the benefits of cross-validation, reducing variance in performance estimates.

### Advantages:
- Faster than K-fold and LOO-CV.
- Reduces variance by repeating the evaluation.

### Disadvantages:
- Can introduce redundancy since the same data points may appear in both the training and test sets in different splits.

## 9.6 What Techniques to Use When

Here are some guidelines for choosing the right evaluation technique:

- **K-fold cross-validation** is the gold standard for small to medium-sized datasets. Typically, **k = 3, 5, or 10**.
- **Train/test split** is useful for large datasets or when computational efficiency is important.
- **Leave-One-Out Cross Validation** is ideal for small datasets but is computationally expensive.
- **Repeated Random Test-Train Splits** offer a balance between speed and variance reduction.

## 9.7 Summary

In this chapter, you learned about various resampling techniques for evaluating the performance of machine learning algorithms:

- **Train and Test Sets**
- **K-fold Cross Validation**
- **Leave One Out Cross Validation**
- **Repeated Random Test-Train Splits**

Each method has its advantages and trade-offs. The best technique depends on factors like dataset size, computational cost, and the need for accurate performance estimates. If in doubt, **10-fold cross-validation** is a safe choice.
