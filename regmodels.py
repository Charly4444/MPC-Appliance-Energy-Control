from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor


# X contains the features, and y is the target variable (Appliances)
data = pd.read_csv('energydata_mod.csv')
X = data.drop('Appliances', axis=1)
y = data['Appliances']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use Min-Max scaling for features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a random forest regressor
model = RandomForestRegressor()
model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

# Visualize training performance
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Appliances Energy Consumption")
plt.ylabel("Predicted Appliances Energy Consumption")
plt.title("Actual vs Predicted Appliances Energy Consumption")
plt.show()

# Save the trained model to a file
joblib.dump(model, 'energy_reg_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
