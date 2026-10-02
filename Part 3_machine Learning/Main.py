import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import mlflow.pyfunc
from datetime import datetime
from contextlib import asynccontextmanager  # <-- Import this for lifespan events

# ==========================================
# 1. LIFESPAN EVENT HANDLER (Replaces on_event)
# ==========================================
# Global holder for our PyFunc model instance
traffic_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles logic that runs on server startup and shutdown."""
    global traffic_model
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MODEL_NAME = "Traffic_Volume_Simplified"
    MODEL_STAGE = "2"

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        print(f"🔄 Fetching streamlined model '{MODEL_NAME}' ({MODEL_STAGE}) from MLflow...")
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        traffic_model = mlflow.pyfunc.load_model(model_uri)
        print("✅ Streamlined model signatures loaded and active!")
    except Exception as e:
        print(f"❌ Critical error loading model: {e}")
        # Server will safely abort startup if the model isn't found
        raise RuntimeError("Microservice initialization aborted due to missing MLflow dependency.")
        
    yield  # Everything before this runs on startup; everything after runs on shutdown
    print("🛑 Cleaning up server state on shutdown...")

# Initialize FastAPI App and pass the lifespan handler
app = FastAPI(
    title="Smart Route Recommendation API - Streamlined",
    version="1.0.0",
    lifespan=lifespan  # <-- Attach lifespan here
)

# ==========================================
# 2. API REQUEST VALIDATION SCHEMA
# ==========================================
class RecommendationRequest(BaseModel):
    is_weekend: int = Field(..., description="0 for Weekday, 1 for Weekend.")
    
    # Updated to pass the example inside json_schema_extra dict
    hourly_weather_forecast: Dict[int, int] = Field(
        ..., 
        description="Dictionary mapping 24 hours to numeric weather codes.",
        json_schema_extra={"example": {14: 1, 15: 1, 16: 2}}
    )
    
    window_size_hours: int = Field(default=3, ge=1, le=12)

# ==========================================
# 3. CONCISE MATRIX CALCULATOR
# ==========================================
def calculate_optimal_timeline(is_weekend: int, weather_forecast: Dict[int, int], window_size: int):
    input_data = []
    
    for hour in range(24):
        weather = weather_forecast.get(hour, weather_forecast.get(str(hour), 0))
        input_data.append({
            'hour': hour,
            'is_weekend': is_weekend,
            'clouds_all': weather
        })
    
    df_input = pd.DataFrame(input_data)
    predicted_delays = traffic_model.predict(df_input)
    
    schedule = []
    for hour in range(24):
        schedule.append({
            'hour': hour,
            'predicted_delay': float(predicted_delays[hour]),
            'weather': weather_forecast.get(hour, weather_forecast.get(str(hour), 0))
        })
        
    schedule_df = pd.DataFrame(schedule)
    
    best_avg_delay = float('inf')
    best_start_hour = 0
    
    for start_hour in range(24 - window_size + 1):
        window_slice = schedule_df.iloc[start_hour : start_hour + window_size]
        avg_delay = window_slice['predicted_delay'].mean()
        
        if avg_delay < best_avg_delay:
            best_avg_delay = avg_delay
            best_start_hour = start_hour
            
    return best_start_hour, best_avg_delay, schedule_df

# ==========================================
# 4. MICROSERVICE ENDPOINT
# ==========================================
@app.post("/recommend-route")
def get_route_recommendation(payload: RecommendationRequest):
    if traffic_model is None:
        raise HTTPException(status_code=503, detail="Model server context uninitialized.")

    start_hour, avg_delay, schedule_df = calculate_optimal_timeline(
        payload.is_weekend, 
        payload.hourly_weather_forecast, 
        payload.window_size_hours
    )
    
    end_hour = start_hour + payload.window_size_hours
    start_time_str = datetime.strptime(f"{start_hour}", "%H").strftime("%I:%M %p")
    end_time_str = datetime.strptime(f"{end_hour}", "%H").strftime("%I:%M %p")
    
    peak_day_delay = schedule_df['predicted_delay'].max()
    time_saved = max(peak_day_delay - avg_delay, 0)
    
    weather_mapping = {0: "Clear Skies", 1: "Rainy", 2: "Heavy Storms"}
    weather_in_window = int(schedule_df.iloc[start_hour:end_hour]['weather'].max())

    plain_language_text = (
        f"Your optimal travel window for today is between {start_time_str} and {end_time_str}. "
        f"By choosing this block, you will experience an average delay of only {round(avg_delay)} minutes, "
        f"saving you roughly {round(time_saved)} minutes compared to peak daily traffic. "
        f"Conditions are expected to be '{weather_mapping.get(weather_in_window, 'Clear Skies')}' during this time."
    )
    
    return {
        "status": "success",
        "optimal_start_hour": start_hour,
        "optimal_end_hour": end_hour,
        "average_delay_minutes": round(avg_delay, 2),
        "estimated_time_saved_minutes": round(time_saved, 2),
        "plain_language_recommendation": plain_language_text
    }
