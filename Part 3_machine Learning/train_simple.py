import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

# 1. Connect to your active local MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Route_Optimization")

# 2. LOAD YOUR PROCESSED CSV DATA 

df = pd.read_csv("data/ml_Metro_Interstate_Traffic_Volume.csv")

# 3. Filter down to the exact 3 features your FastAPI app expects
# Ensure these match the exact column names present in your CSV file!
features = ['hour', 'is_weekend', 'clouds_all']
X = df[features]
y = df['traffic_volume']  

# 4. Train the model
print("⚡ Training the simplified model...")
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)

# 5. Push the model into the placeholder slot you created in the UI
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Traffic_Volume_Simplified" # Matches your UI slot exactly
    )
    print("✅ Model successfully pushed to MLflow UI!")
