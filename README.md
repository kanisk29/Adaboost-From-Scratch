# Manual AdaBoost Implementation using Decision Stumps

This project demonstrates a **manual implementation of the AdaBoost algorithm** using **Decision Trees (Decision Stumps)** in Python. Instead of using a built-in AdaBoost library, the boosting process is implemented step-by-step to show how the algorithm works internally.

The implementation includes weight updates, resampling based on probability ranges, and sequential training of weak learners.

---

## Overview

AdaBoost (Adaptive Boosting) is an ensemble learning algorithm that combines multiple weak learners to create a strong classifier.

The core idea:

1. Train a weak learner (decision stump).
2. Identify misclassified samples.
3. Increase weights of misclassified samples.
4. Decrease weights of correctly classified samples.
5. Resample the dataset based on the updated weights.
6. Train another weak learner on the new dataset.
7. Repeat the process multiple times.

Each weak learner contributes to the final prediction with a weight called **alpha**, which depends on its error.

---

## Algorithm Steps Implemented

### 1. Initialize Weights

All samples start with equal weights.

[
w_i = \frac{1}{N}
]

where

* (N) = number of samples

---

### 2. Train Weak Learner

A **Decision Tree with max depth = 1** (decision stump) is trained.

```python
DecisionTreeClassifier(max_depth=1)
```

---

### 3. Compute Weighted Error

[
error = \sum w_i \quad \text{for misclassified samples}
]

---

### 4. Compute Alpha (Learner Weight)

[
\alpha = \frac{1}{2} \log \left(\frac{1-error}{error}\right)
]

This determines how much influence the learner has in the final model.

---

### 5. Update Sample Weights

Misclassified samples get **higher weights** while correctly classified samples get **lower weights**.

[
w_i =
\begin{cases}
w_i \times e^{\alpha} & \text{if misclassified} \
w_i \times e^{-\alpha} & \text{if correct}
\end{cases}
]

---

### 6. Normalize Weights

Weights are normalized so that they sum to 1.

[
w_i = \frac{w_i}{\sum w_i}
]

---

### 7. Create Probability Ranges

Using cumulative sums to create sampling ranges.

```python
df["range_upper"] = np.cumsum(df["new_weights"])
df["range_lower"] = np.cumsum(df["new_weights"]) - df["new_weights"]
```

These ranges help simulate **weighted sampling**.

---

### 8. Resampling

Random numbers between **0 and 1** are generated.

Each random number falls into one of the probability ranges, determining which row is sampled.

```python
row = df[(i < df["range_upper"]) & (i > df["range_lower"])].index[0]
```

Rows with **higher weights appear more frequently** in the next dataset.

---

### 9. Train Next Weak Learner

The resampled dataset is used to train another **decision stump**.

The process repeats for multiple rounds.

---

## Functions in the Implementation

### `trainer(df)`

Trains a decision stump and generates predictions.

### `update_weights(df)`

Calculates error, alpha, and updates sample weights.

### `assign_range(df)`

Creates cumulative probability ranges for sampling.

### `assigned(df)`

Performs weighted sampling using random numbers.

---

## Libraries Used

* **scikit-learn** – Decision Tree classifier
* **pandas** – Data handling
* **numpy** – Mathematical operations
* **mlxtend** – Decision boundary visualization
* **seaborn** – Visualization support

---

## Visualization

Decision boundaries are plotted using:

```python
plot_decision_regions()
```

This helps visualize how each decision stump separates the data.

---

## Key Learning Points

* How AdaBoost updates weights after each learner
* Why misclassified samples become more important
* How weighted resampling works
* How weak learners combine to form a strong model

---

## How to Run

1. Clone the repository

```bash
git clone https://github.com/your-username/manual-adaboost
```

2. Install dependencies

```bash
pip install scikit-learn pandas numpy seaborn mlxtend
```

3. Run the script

```bash
python adaboost_manual.py
```

---
