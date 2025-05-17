import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Sample dataset structure (replace with real-time data source)
data = pd.read_csv('sandstorm_data.csv')  # Columns: wind_speed, humidity, dust_concentration, temperature, label

# Preprocessing
X = data[['wind_speed', 'humidity', 'dust_concentration', 'temperature']]
y = data['label']  # 0 = No Sandstorm, 1 = Sandstorm

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Prediction Accuracy: {accuracy * 100:.2f}%")

# Predict on new data
def predict_sandstorm(wind_speed, humidity, dust_concentration, temperature):
    result = model.predict([[wind_speed, humidity, dust_concentration, temperature]])
    return "Sandstorm Expected" if result[0] == 1 else "No Sandstorm"

# Example
print(predict_sandstorm(45, 20, 180, 38))