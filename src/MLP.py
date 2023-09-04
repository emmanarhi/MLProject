import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score

# Data
df = pd.read_csv("../data/Maternal Health Risk Data Set.csv")

# Do the same things as in the project but keep the original three classes
df["RiskLevel"].replace({"high risk": 1, "mid risk": 1, "low risk": 0}, inplace=True)

y = df["RiskLevel"].to_numpy()
X = df.drop(["RiskLevel"], axis=1).to_numpy()
print(df["RiskLevel"].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=42)

# I will skip standardizing the data because neural networks solve this issue with
# activation functions

clf = MLPClassifier(random_state=42)

clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: ", accuracy)
