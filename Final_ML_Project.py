#Using customer retail dataset implement various models and compare them

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

#Load customer dataset using Pandas
data = pd.read_csv("customer_retail_dataset.csv")

#Use required features
#Required columns: Quantity, UnitPrice, Country
data = data[["Quantity", "UnitPrice", "Country"]]

#Handle missing values
data = data.dropna()

#Encode categorical column using LabelEncoder
label_encoder = LabelEncoder()
data["Country"] = label_encoder.fit_transform(data["Country"])

#Create a target column for supervised learning
#HighValuePurchase = 1 if total purchase value is higher than the median, otherwise 0
data["TotalAmount"] = data["Quantity"] * data["UnitPrice"]
median_amount = data["TotalAmount"].median()
data["HighValuePurchase"] = data["TotalAmount"].apply(lambda x: 1 if x > median_amount else 0)

#Define features and target
X = data[["Quantity", "UnitPrice", "Country"]]
y = data["HighValuePurchase"]

#Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

#Train Decision Tree model
decision_tree_model = DecisionTreeClassifier(random_state=42)
decision_tree_model.fit(X_train, y_train)
decision_tree_predictions = decision_tree_model.predict(X_test)

#Train KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

#Evaluate models using Accuracy and Confusion Matrix
logistic_accuracy = accuracy_score(y_test, logistic_predictions)
decision_tree_accuracy = accuracy_score(y_test, decision_tree_predictions)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("Logistic Regression Accuracy:", round(logistic_accuracy, 4))
print("Decision Tree Accuracy:", round(decision_tree_accuracy, 4))
print("KNN Accuracy:", round(knn_accuracy, 4))

print("\nLogistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, decision_tree_predictions))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))

#Customer Distribution Graph
plt.figure(figsize=(10, 5))
data["Country"].value_counts().plot(kind="bar")
plt.title("Customer Distribution by Country")
plt.xlabel("Encoded Country")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("customer_distribution.png")
plt.show()

#Model Accuracy Comparison Graph
models = ["Logistic Regression", "Decision Tree", "KNN"]
accuracies = [logistic_accuracy, decision_tree_accuracy, knn_accuracy]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies)
plt.title("Model Accuracy Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("model_accuracy_comparison.png")
plt.show()

#Show final comparison Table
results = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracies
})
print("\nModel Accuracy Comparison")
print(results)

best_model = results.loc[results["Accuracy"].idxmax(), "Model"]
print("\nBest Performing Model:", best_model)
