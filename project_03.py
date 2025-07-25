# Essential libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
sb.set(style="darkgrid")

# Load the dataset
car_df = pd.read_csv('car data.csv')
print(car_df.head())

# Dataset info
car_df.info()

# Missing values
print("\nMissing Values:\n", car_df.isnull().sum())

# Statistical summary
print("\nDataset Description:\n", car_df.describe())

# Display unique categories
print("Available Fuel Types:", car_df['Fuel_Type'].unique())
print("Transmission Variants:", car_df['Transmission'].unique())
print("Sales Categories:", car_df['Selling_type'].unique())
print("Owner Categories:", car_df['Owner'].unique())

# Encode categorical variables
df_ready = pd.get_dummies(car_df, drop_first=True)
print(df_ready.head())

# Create multiple figures (updated visuals only)
figures = []

# Selling Price Distribution (Changed style and color)
fig1 = plt.figure(figsize=(8, 5))
plt.hist(car_df['Selling_Price'], bins=12, color='tomato', edgecolor='black', alpha=0.8)
plt.title('Distribution of Selling Price', fontsize=14, fontweight='bold')
plt.xlabel('Selling Price (in Lakhs)')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
figures.append(fig1)

# Present Price vs Selling Price Scatterplot (changed markers and style)
fig2 = plt.figure(figsize=(8, 5))
fuel_palette = {'Petrol': '#1f77b4', 'Diesel': '#ff7f0e', 'CNG': '#2ca02c'}
for fuel_type in car_df['Fuel_Type'].unique():
    subset = car_df[car_df['Fuel_Type'] == fuel_type]
    plt.scatter(subset['Present_Price'], subset['Selling_Price'], 
                label=fuel_type, s=70, alpha=0.7, edgecolor='black')

plt.title('Selling vs Present Price by Fuel', fontsize=14, fontweight='bold')
plt.xlabel('Present Price (Lakhs)')
plt.ylabel('Selling Price (Lakhs)')
plt.legend(title='Fuel Type')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
figures.append(fig2)

# Boxplot of Fuel Type vs Selling Price (changed style and orientation)
fig3 = plt.figure(figsize=(8, 5))
sb.boxplot(
    data=car_df,
    y='Fuel_Type',
    x='Selling_Price',
    palette='Set3'
)
plt.title('Selling Price across Fuel Types', fontsize=14, fontweight='bold')
plt.ylabel('Fuel Type')
plt.xlabel('Selling Price (Lakhs)')
plt.grid(True, linestyle=':', alpha=0.4)
plt.tight_layout()
figures.append(fig3)

# Model data preparation
model_df = car_df.drop('Car_Name', axis=1)
model_df = pd.get_dummies(model_df, drop_first=True)

X_data = model_df.drop('Selling_Price', axis=1)
y_data = model_df['Selling_Price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=10
)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Regression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_prediction = regressor.predict(X_test)

r2_val = r2_score(y_test, y_prediction)
mae_val = mean_absolute_error(y_test, y_prediction)

print(f"Model R2 Score: {r2_val}")
print(f"Mean Absolute Error: {mae_val}")

# Prediction vs Actual (changed color, added regression line)
fig4 = plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_prediction, color='purple', alpha=0.6, label='Predictions')
z = np.polyfit(y_test, y_prediction, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--", label='Trendline')
plt.plot([y_data.min(), y_data.max()], [y_data.min(), y_data.max()], 'g--', label='Ideal Fit')
plt.xlabel("Actual Price (Lakhs)")
plt.ylabel("Predicted Price (Lakhs)")
plt.title("Actual vs Predicted Selling Price")
plt.legend()
plt.grid(True, linestyle='-.', alpha=0.4)
plt.tight_layout()
figures.append(fig4)

# Show all figures at once
plt.show()
