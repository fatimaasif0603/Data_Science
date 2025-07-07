import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Pakistan_Poverty_DS.csv')
print(data.columns.tolist())  # Verify column names

# Plot GDP Growth (replace with your target column)
plt.figure(figsize=(10, 5))
plt.plot(data['Year'], data['GDP Growth Rate (%)'], marker='o', color='purple')
plt.title('GDP Growth Over Time')
plt.xlabel('Year')
plt.ylabel('GDP Growth (%)')
plt.grid(True)
plt.show()