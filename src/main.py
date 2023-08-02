import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, log_loss, zero_one_loss
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# dataset
df = pd.read_csv("data/Maternal Health Risk Data Set.csv")

# Change labels from strings to integers
df["RiskLevel"].replace({"high risk": 1, "mid risk": 1, "low risk": 0}, inplace=True)

# Assigning features and labels
y = df["RiskLevel"].to_numpy()
X = df.drop(["RiskLevel"], axis=1).to_numpy()

# Checking the classes
print(df["RiskLevel"].value_counts())

# Splitting the data into training, validation and test sets
X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=0.6, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, shuffle=True, random_state=42)

# Standardizing the data
scaler = StandardScaler()
# Fitting the scaler to the training set and then using it on other sets
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Logistic Regression

log_reg = LogisticRegression()
log_reg = log_reg.fit(X_train, y_train)

y_pred_train_l = log_reg.predict(X_train)
y_pred_val_l = log_reg.predict(X_val)

loss_train_l = log_loss(y_pred_train_l, y_train)
loss_val_l = log_loss(y_pred_val_l, y_val)
f1_l = f1_score(y_val, y_pred_val_l)

print("Logistic Regression\n")
print("Training error:", loss_train_l, "\nValidation Error:", loss_val_l, "\nF1-score:", f1_l)


# K-nearest neighbors

neigh = KNeighborsClassifier(n_neighbors=6)
neigh = neigh.fit(X_train, y_train)

y_pred_train_k = neigh.predict(X_train)
y_pred_val_k = neigh.predict(X_val)

loss_train_k = zero_one_loss(y_pred_train_k, y_train)
loss_val_k = zero_one_loss(y_pred_val_k, y_val)
f1_k = f1_score(y_val, y_pred_val_k)

print("\nK-nearest neighbors\n")
print("Training error:", loss_train_k, "\nValidation error:", loss_val_k, "\nF1-score:", f1_k)

# Because I didn't fine-tune, here are the test set results:

y_pred_test_l = log_reg.predict(X_test)
y_pred_test_k = neigh.predict(X_test)

loss_test_l = log_loss(y_pred_test_l, y_test)
loss_test_k = zero_one_loss(y_pred_test_k, y_test)

f1_test_k = f1_score(y_test, y_pred_test_l)
f1_test_l = f1_score(y_test, y_pred_test_k)

print("\nTest sets:\n")
print("Logistic Regression: Test error:", loss_test_l, ", F1-score:", f1_test_l)
print("KNN: Test error:", loss_test_k, ", F1-score:", f1_test_k)

# Display the results as confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

sns.heatmap(confusion_matrix(y_val, y_pred_val_l), annot=True, fmt='g', ax=axes[0, 0])
axes[0, 0].set_title("Logistic Regression Confusion Matrix with the validation set")
axes[0, 0].set_xlabel("Predicted labels", fontsize=15)
axes[0, 0].set_ylabel("True labels", fontsize=15)

sns.heatmap(confusion_matrix(y_val, y_pred_val_k), annot=True, fmt='g', ax=axes[0, 1])
axes[0, 1].set_title("KNN Confusion Matrix with the validation set")
axes[0, 1].set_xlabel("Predicted labels", fontsize=15)
axes[0, 1].set_ylabel("True labels", fontsize=15)

sns.heatmap(confusion_matrix(y_test, y_pred_test_l), annot=True, fmt='g', ax=axes[1, 0])
axes[1, 0].set_title("Logistic Regression Confusion Matrix with the test set")
axes[1, 0].set_xlabel("Predicted labels", fontsize=15)
axes[1, 0].set_ylabel("True labels", fontsize=15)

sns.heatmap(confusion_matrix(y_test, y_pred_test_k), annot=True, fmt='g', ax=axes[1, 1])
axes[1, 1].set_title("KNN Confusion Matrix with the test set")
axes[1, 1].set_xlabel("Predicted labels", fontsize=15)
axes[1, 1].set_ylabel("True labels", fontsize=15)

plt.tight_layout()
plt.show()
