# 🪻 Iris Species Predictor: A Machine Learning Approach
# Author: Fatima Asif

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, 
                            confusion_matrix,
                            ConfusionMatrixDisplay)

# Load dataset with enhanced features
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, 
                      columns=[col.replace(' (cm)', '') for col in iris.feature_names])
iris_df['species'] = iris.target_names[iris.target]

# Feature Engineering
iris_df['sepal_ratio'] = iris_df['sepal length'] / iris_df['sepal width']
iris_df['petal_ratio'] = iris_df['petal length'] / iris_df['petal width']

# ========== Enhanced Visualization ==========
plt.style.use('seaborn-v0_8-pastel')

# 1. Pairplot with new ratios
sns.pairplot(iris_df, hue='species', 
             vars=['sepal length', 'sepal width', 
                  'petal length', 'petal width',
                  'sepal_ratio', 'petal_ratio'],
             palette='husl')
plt.suptitle("Multivariate Analysis of Iris Features", y=1.02)
plt.show()

# 2. Violin Plots for feature distribution
plt.figure(figsize=(12, 8))
for i, feature in enumerate(['sepal length', 'sepal width', 
                            'petal length', 'petal width']):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=iris_df, 
                  palette='Set2', inner='quartile')
    plt.title(f"{feature.title()} Distribution")
plt.tight_layout()
plt.show()

# ========== Advanced Modeling ==========
X = iris_df.drop(['species'], axis=1)
y = iris.target

# Define candidate models with tuned parameters
models = {
    'Support Vector Machine': make_pipeline(
        RobustScaler(),
        SVC(kernel='rbf', C=1.0, gamma='scale')
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        max_depth=3,
        random_state=42
    ),
    'Gaussian Naive Bayes': GaussianNB()
}

# Model Evaluation Framework
results = []
for name, model in models.items():
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results.append({
        'Model': name,
        'Mean Accuracy': cv_scores.mean(),
        'Std Dev': cv_scores.std()
    })
    print(f"{name} Cross-validation Accuracy: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Convert results to DataFrame
results_df = pd.DataFrame(results).sort_values('Mean Accuracy', ascending=False)
print("\nModel Performance Summary:")
print(results_df.to_string(index=False))

# Train best model
best_model = models[results_df.iloc[0]['Model']]
best_model.fit(X, y)

# ========== Prediction Demo ==========
demo_samples = [
    [5.1, 3.5, 1.4, 0.2, 5.1/3.5, 1.4/0.2],  # Setosa
    [6.3, 2.8, 4.9, 1.5, 6.3/2.8, 4.9/1.5],   # Versicolor
    [7.2, 3.6, 6.1, 2.5, 7.2/3.6, 6.1/2.5]    # Virginica
]

print("\nPrediction Examples:")
for sample in demo_samples:
    pred = best_model.predict([sample])[0]
    print(f"Features: {sample[:4]} → Predicted: {iris.target_names[pred]}")

# ========== Model Interpretation ==========
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    fi_df = fi_df.sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=fi_df, palette='rocket')
    plt.title("Feature Importance Analysis")
    plt.show()

# Confusion Matrix Visualization
y_pred = best_model.predict(X)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=iris.target_names)
disp.plot(cmap='Blues')
plt.title("Classification Performance")
plt.show()