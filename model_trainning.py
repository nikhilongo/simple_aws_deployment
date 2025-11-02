import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Load dataset
df = pd.read_csv("placement.csv")

# Separate target and features
y = df["package"]
X = df.drop(columns=["package"])

# Scale CGPA only
scaler = StandardScaler()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale CGPA column
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled["cgpa"] = scaler.fit_transform(X_train[["cgpa"]])
X_test_scaled["cgpa"] = scaler.transform(X_test[["cgpa"]])

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluate model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R² Score:", r2)
print("Mean Squared Error:", mse)

# Save model
with open("linear_regression_package.pkl", "wb") as f:
    pickle.dump(model, f)

# Save scaler
with open("cgpa_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Model and scaler saved successfully.")
