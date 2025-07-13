#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')

#reading the csv file
economic_data = pd.read_csv("Pakistan_Poverty_DS.csv")

#basic info about the dataset
print("First 5 records:")
print(economic_data.head())
print("\nLast 5 records:")
print(economic_data.tail())
print("\nDataset information:")
print(economic_data.info())
print("\nStatistical summary:")
print(economic_data.describe())

#checking for any duplicates
has_duplicates_year = economic_data["Year"].duplicated().any()
print("\nDuplicate years check:", has_duplicates_year)

#visualization 
#1. Economic trend line plots
fig, axs1 = plt.subplots(2, 2)
fig.suptitle("Pakistan Economic Indicators (2000-2023)", fontsize=14, y=1.02)

# Unemployment trend
x_data = economic_data['Year']
y_data = economic_data["Unemployment Rate (%)"]
axs1[0,0].plot(x_data, y_data, c="#4CAF50", linewidth=2.5)
axs1[0,0].set_xlabel("Year")
axs1[0,0].set_ylabel("Rate (%)")
axs1[0,0].set_title("Unemployment Trend")

# Poverty trend
y_data = economic_data["Poverty Headcount Ratio (%)"]
axs1[0,1].plot(x_data, y_data, c="#FF5722", linewidth=2.5)
axs1[0,1].set_xlabel("Year")
axs1[0,1].set_ylabel("Ratio (%)")
axs1[0,1].set_title("Poverty Level Trend")

# GDP growth trend
y_data = economic_data["GDP Growth Rate (%)"]
axs1[1,0].plot(x_data, y_data, c="#2196F3", linewidth=2.5)
axs1[1,0].set_xlabel("Year")
axs1[1,0].set_ylabel("Growth Rate (%)")
axs1[1,0].set_title("GDP Growth Trend")

# Government spending trend
y_data = economic_data["Government Social Spending (% of GDP)"]
axs1[1,1].plot(x_data, y_data, c="#9C27B0", linewidth=2.5)
axs1[1,1].set_xlabel("Year")
axs1[1,1].set_ylabel("Spending (% of GDP)")
axs1[1,1].set_title("Social Spending Trend")

#2. Comparative analysis plots
# Multiline comparison plot
plt.figure(figsize=(10, 6))
plt.title("Key Economic Indicators Comparison", pad=20)
y_data = economic_data["Unemployment Rate (%)"]
plt.plot(x_data, y_data, c="#4CAF50", label="Unemployment rate", linewidth=2.5)
y_data = economic_data["GDP Growth Rate (%)"]
plt.plot(x_data, y_data, c="#2196F3", label="GDP growth rate", linewidth=2.5)
y_data = economic_data["Agriculture Growth Rate (%)"]
plt.plot(x_data, y_data, c="#FFC107", label="Agriculture growth", linewidth=2.5)
y_data = economic_data["Government Social Spending (% of GDP)"]
plt.plot(x_data, y_data, c="#9C27B0", label="Social spending", linewidth=2.5)

plt.xlabel("Year")
plt.ylabel("Percentage")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Climate impact analysis
fig, ax1 = plt.subplots(figsize=(10, 6))
plt.title("Economic Impact of 2010 Floods", pad=20)

climate = economic_data["Climate Disasters (count)"]
unemployment = economic_data["Unemployment Rate (%)"]
poverty = economic_data["Poverty Headcount Ratio (%)"]

# First Y-axis (for %)
ax1.plot(x_data, unemployment, label="Unemployment Rate", color="#4CAF50", linewidth=2.5)
ax1.plot(x_data, poverty, label="Poverty Ratio", color="#FF5722", linewidth=2.5)
ax1.set_xlabel("Year")
ax1.set_ylabel("Economic Indicators (%)", color='white')
ax1.tick_params(axis='y', labelcolor='white')

# Vertical line at 2010
ax1.axvline(x=2010, color='white', linestyle='--', linewidth=1.5)
ax1.text(2010, unemployment.max(), "← 2010 Floods", color='white', fontsize=10)

# Second Y-axis (for climate disasters count)
ax2 = ax1.twinx()
ax2.plot(x_data, climate, label="Climate Disasters", color="#00BCD4", linewidth=2.5)
ax2.set_ylabel("Climate Disasters Count", color='white')
ax2.tick_params(axis='y', labelcolor='white')

# Merge legends from both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

#3. Unemployment bar chart
plt.figure(figsize=(10, 6))
plt.title("Unemployment Rate Over Years", pad=20)
y_data = economic_data["Unemployment Rate (%)"]
plt.bar(x_data, y_data, color="#4CAF50", edgecolor='#2E7D32')
plt.xlabel("Year")
plt.ylabel("Unemployment Rate (%)")

#4. Statistical distribution plots
fig, axs2 = plt.subplots(2, 2)
fig.suptitle("Economic Indicators Distribution", fontsize=14, y=1.02)

boxprops = dict(facecolor="#4CAF50", edgecolor='#2E7D32')
axs2[0,0].boxplot(economic_data["Unemployment Rate (%)"], 
                 tick_labels=["Unemployment"], 
                 patch_artist=True,
                 boxprops=boxprops)
axs2[0,1].boxplot(economic_data["GDP Growth Rate (%)"], 
                 tick_labels=["GDP growth"], 
                 patch_artist=True,
                 boxprops=dict(facecolor="#2196F3", edgecolor='#1565C0'))
axs2[1,0].boxplot(economic_data["Government Social Spending (% of GDP)"], 
                 tick_labels=["Social spending"], 
                 patch_artist=True,
                 boxprops=dict(facecolor="#9C27B0", edgecolor='#6A1B9A'))
axs2[1,1].boxplot(economic_data["Poverty Headcount Ratio (%)"], 
                 tick_labels=["Poverty ratio"], 
                 patch_artist=True,
                 boxprops=dict(facecolor="#FF5722", edgecolor='#D84315'))

for ax in axs2.flat:
    ax.set_ylabel("Percentage")

#5. Unemployment distribution histogram
plt.figure(figsize=(10, 6))
plt.title("Unemployment Rate Distribution", pad=20)
plt.hist(economic_data["Unemployment Rate (%)"], bins=10, color="#4CAF50", edgecolor='#2E7D32')
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")

#6. Relationship analysis scatter plots
fig, axs3 = plt.subplots(2, 2)
fig.suptitle("Economic Relationships Analysis", fontsize=14, y=1.02)

x_data = economic_data["Unemployment Rate (%)"]

# Unemployment vs GDP Growth
y_data = economic_data["GDP Growth Rate (%)"]
axs3[0,0].scatter(x_data, y_data, c="#2196F3")
m, b = np.polyfit(x_data, y_data, 1)
axs3[0,0].plot(x_data, m*x_data + b, color="#FFC107", linestyle='--', label="Trend")
axs3[0,0].set_xlabel("Unemployment Rate")
axs3[0,0].set_ylabel("GDP Growth Rate")
axs3[0,0].legend()

# Unemployment vs Inflation
y_data = economic_data["Inflation Rate (%)"]
axs3[0,1].scatter(x_data, y_data, c="#FF5722")
m, b = np.polyfit(x_data, y_data, 1)
axs3[0,1].plot(x_data, m*x_data + b, color="#FFC107", linestyle='--', label="Trend")
axs3[0,1].set_xlabel("Unemployment Rate")
axs3[0,1].set_ylabel("Inflation Rate")
axs3[0,1].legend()

# Unemployment vs Climate Disasters
y_data = economic_data["Climate Disasters (count)"]
axs3[1,0].scatter(x_data, y_data, c="#00BCD4")
m, b = np.polyfit(x_data, y_data, 1)
axs3[1,0].plot(x_data, m*x_data + b, color="#FFC107", linestyle='--', label="Trend")
axs3[1,0].set_xlabel("Unemployment Rate")
axs3[1,0].set_ylabel("Climate Disasters")
axs3[1,0].legend()

# Unemployment vs Agriculture Growth
y_data = economic_data["Agriculture Growth Rate (%)"]
axs3[1,1].scatter(x_data, y_data, c="#8BC34A")
m, b = np.polyfit(x_data, y_data, 1)
axs3[1,1].plot(x_data, m*x_data + b, color="#FFC107", linestyle='--', label="Trend")
axs3[1,1].set_xlabel("Unemployment Rate")
axs3[1,1].set_ylabel("Agriculture Growth")
axs3[1,1].legend()

# COVID impact analysis
before_covid = economic_data[economic_data["Year"] < 2020]
after_covid = economic_data[economic_data["Year"] >= 2020]
print("\nCOVID-19 Impact Analysis:")
print(f"Average Unemployment before 2020: {before_covid['Unemployment Rate (%)'].mean():.2f}%")
print(f"Average Unemployment after 2020: {after_covid['Unemployment Rate (%)'].mean():.2f}%")

plt.figure(figsize=(8, 6))
plt.title("Unemployment Before and After COVID-19", pad=20)
labels = ['Before COVID (2000-2019)', 'After COVID (2020-2023)']
values = [
    before_covid["Unemployment Rate (%)"].mean(),
    after_covid["Unemployment Rate (%)"].mean()
]
plt.bar(labels, values, color=["#4CAF50", "#FF5722"], edgecolor=['#2E7D32', '#D84315'])
plt.ylabel("Unemployment Rate (%)")

# Correlation analysis
corr_matrix = economic_data.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.matshow(corr_matrix, cmap="coolwarm")

# Add color bar
fig.colorbar(cax)

# Add tick labels
ax.set_xticks(np.arange(len(corr_matrix.columns)))
ax.set_yticks(np.arange(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90)
ax.set_yticklabels(corr_matrix.columns)

# Annotate with correlation values
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        value = corr_matrix.iloc[i, j]
        ax.text(j, i, f"{value:.2f}", ha='center', va='center',
                color='white' if abs(value) > 0.5 else 'black', fontsize=9)

plt.title("Economic Indicators Correlation Matrix", pad=20)

plt.tight_layout()
plt.show()