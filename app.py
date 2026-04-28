import gradio as gr
import yfinance as yf
import numpy as np
import pandas as pd
from keras.models import load_model
import joblib
import os

# Load the saved components
# Using a check to ensure the app doesn't crash during build if files are missing
MODEL_PATH = 'stock_model.keras'
SCALER_PATH = 'scaler.gz'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None

def predict_next_day(ticker):
    if model is None or scaler is None:
        return "Error: Model or Scaler files not found in the repository."
    
    try:
        # 1. Fetch data
        # We use multi_level_index=False to fix the common yfinance MultiIndex error
        df = yf.download(ticker, period='90d', interval='1d', multi_level_index=False)
        
        if df.empty: 
            return f"Error: No data found for ticker '{ticker}'."

        # 2. Re-create technical indicators
        df['MA7'] = df['Close'].rolling(window=7).mean()
        df['MA21'] = df['Close'].rolling(window=21).mean()
        df.dropna(inplace=True)

        # 3. Prepare the sliding window (Last 60 days)
        # Ensure we have enough rows after dropping NaNs
        if len(df) < 60:
            return "Error: Not enough historical data to generate a 60-day window."

        features = df[['Close', 'MA7', 'MA21']].tail(60).values
        
        # 4. Scale and Reshape
        scaled_features = scaler.transform(features)
        # Reshape to (1, 60, 3) -> 1 sample, 60 timesteps, 3 features
        input_data = np.reshape(scaled_features, (1, 60, 3))

        # 5. Predict
        prediction_scaled = model.predict(input_data, verbose=0)
        
        # 6. Inverse Transform
        # Since the scaler was fit on 3 columns, we create a dummy with 3 cols
        dummy = np.zeros((1, 3))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction_final = scaler.inverse_transform(dummy)[0, 0]

        return f"Predicted Next Closing Price for {ticker}: ₹{prediction_final:.2f}"
    
    except Exception as e:
        return f"Technical Error: {str(e)}"

# Professional Gradio Interface
interface = gr.Interface(
    fn=predict_next_day,
    inputs=gr.Textbox(
        label="Stock Ticker", 
        placeholder="Enter Ticker (e.g., TCS.NS for NSE, AAPL for NASDAQ)..."
    ),
    outputs=gr.Textbox(label="Forecasted Price"),
    title="📈 StockPulse: LSTM Market Predictor",
    description="This AI model uses Long Short-Term Memory (LSTM) networks to predict the next day's closing price based on a 60-day window of historical prices and Moving Averages.",
    theme="soft"
)

if __name__ == "__main__":
    interface.launch()