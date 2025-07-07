# Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X = iris.data  # Features (sepal/petal measurements)
y = iris.target  # Target (species: 0, 1, 2)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict and check accuracy
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# Example prediction
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Sample measurements
predicted_species = model.predict(new_flower)
print(f"Predicted species: {iris.target_names[predicted_species][0]}")