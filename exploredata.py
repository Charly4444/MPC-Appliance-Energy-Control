import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
url = "KAG_energydata_complete.csv"
df = pd.read_csv(url, parse_dates=['date'])

# Drop the 'date' column
df = df.drop('date', axis=1)

# Visualize correlations with the target variable
correlation_with_target = df.corr()['Appliances'].sort_values(ascending=False)
print("Correlation with Appliances:\n", correlation_with_target)

# correlation threshold for feature selection
correlation_threshold = 0.08

# Select features with correlation greater than the threshold
selected_features = correlation_with_target[abs(correlation_with_target) > correlation_threshold].index

# Create a DataFrame with selected features
df_selected = df[selected_features]

# Save the data to a new CSV file
df_selected.to_csv('energydata_mod.csv', index=False)

# Visualize correlations among selected features
correlation_matrix_selected = df_selected.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix_selected, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for Selected Features')
plt.show()

# Display the selected features
print("Selected Features:")
print(df_selected.columns)
