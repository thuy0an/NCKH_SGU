# 📊 Chapter 7: Prepare Your Data for Machine Learning

## 🔍 Why Preprocess Data?

Most machine learning algorithms make assumptions about the input data. Proper preprocessing:

- Ensures all features are on the same scale.
- Helps expose the structure of the problem.
- Can improve the performance of models.

⚠️ Different algorithms may require different preprocessing techniques.

You should try multiple transformations and test which combination of preprocessing and model gives the best result.

---

## ⚙️ Common Data Transformation Techniques

In this chapter, we explore 4 common data preprocessing techniques using `scikit-learn`, all demonstrated on the **Pima Indians Diabetes Dataset**.

Each example involves:
1. Loading the dataset.
2. Splitting it into input (X) and output (y).
3. Applying the transformation.
4. Printing the first few rows of transformed data.

---

### 1️⃣ Rescale Data (Min-Max Normalization)

**Purpose:** Rescale features to a specific range, typically [0, 1].

**When to use:** 
- Features with different ranges.
- Algorithms that rely on distance (e.g., KNN) or gradient descent (e.g., Neural Networks).

**Tool:** `MinMaxScaler`

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_rescaled = scaler.fit_transform(X)
