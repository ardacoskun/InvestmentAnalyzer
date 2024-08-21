from flask import Flask, request, jsonify
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
from core.data_fetcher import fetch_stock_data
from core.analysis import (
    calculate_moving_averages, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_stochastic, calculate_atr,
    calculate_momentum, calculate_buy_or_sell_signal
)
import threading

# Flask Uygulaması
flask_app = Flask(__name__)

# Eğitilmiş modeli yükleme
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@flask_app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    tickers = data.get('tickers', [])
    
    buy_opportunities = []
    sell_opportunities = []

    for ticker in tickers:
        stock_data = fetch_stock_data(ticker)

        if stock_data is None:
            continue

        short_sma, long_sma = calculate_moving_averages(stock_data['Close'])
        rsi = calculate_rsi(stock_data['Close'])
        macd, signal_line = calculate_macd(stock_data['Close'])
        upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
        stochastic = calculate_stochastic(stock_data['Close'])
        atr = calculate_atr(stock_data['Close'], stock_data['High'], stock_data['Low'])
        momentum = calculate_momentum(stock_data['Close'])

        feature_values = {
            'rsi': rsi.iloc[-1],
            'macd': macd.iloc[-1],
            'stochastic': stochastic.iloc[-1],
            'momentum': momentum.iloc[-1],
            'short_sma': short_sma.iloc[-1],
            'long_sma': long_sma.iloc[-1],
            'upper_band': upper_band.iloc[-1],
            'lower_band': lower_band.iloc[-1],
            'atr': atr.iloc[-1]
        }

        features = pd.DataFrame([feature_values])
        signal = model.predict(features)[0]

        if signal == "buy":
            buy_opportunities.append(ticker)
        elif signal == "sell":
            sell_opportunities.append(ticker)

    return jsonify({
        "buy_opportunities": buy_opportunities,
        "sell_opportunities": sell_opportunities
    })

# FastAPI Uygulaması
fastapi_app = FastAPI()

class SignalRequest(BaseModel):
    stock_symbol: str

@fastapi_app.post("/buy-signal/")
def get_buy_signal(request: SignalRequest):
    stock_data = fetch_stock_data(request.stock_symbol)
    if stock_data is None:
        return {"signal": "no data", "symbol": request.stock_symbol}

    short_sma, long_sma = calculate_moving_averages(stock_data['Close'])
    rsi = calculate_rsi(stock_data['Close'])
    macd, signal_line = calculate_macd(stock_data['Close'])
    upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
    stochastic = calculate_stochastic(stock_data['Close'])
    atr = calculate_atr(stock_data['Close'], stock_data['High'], stock_data['Low'])
    momentum = calculate_momentum(stock_data['Close'])

    feature_values = {
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'stochastic': stochastic.iloc[-1],
        'momentum': momentum.iloc[-1],
        'short_sma': short_sma.iloc[-1],
        'long_sma': long_sma.iloc[-1],
        'upper_band': upper_band.iloc[-1],
        'lower_band': lower_band.iloc[-1],
        'atr': atr.iloc[-1]
    }

    features = pd.DataFrame([feature_values])
    signal = model.predict(features)[0]

    return {"signal": signal, "symbol": request.stock_symbol}

@fastapi_app.post("/sell-signal/")
def get_sell_signal(request: SignalRequest):
    stock_data = fetch_stock_data(request.stock_symbol)
    if stock_data is None:
        return {"signal": "no data", "symbol": request.stock_symbol}

    short_sma, long_sma = calculate_moving_averages(stock_data['Close'])
    rsi = calculate_rsi(stock_data['Close'])
    macd, signal_line = calculate_macd(stock_data['Close'])
    upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
    stochastic = calculate_stochastic(stock_data['Close'])
    atr = calculate_atr(stock_data['Close'], stock_data['High'], stock_data['Low'])
    momentum = calculate_momentum(stock_data['Close'])

    feature_values = {
        'rsi': rsi.iloc[-1],
        'macd': macd.iloc[-1],
        'stochastic': stochastic.iloc[-1],
        'momentum': momentum.iloc[-1],
        'short_sma': short_sma.iloc[-1],
        'long_sma': long_sma.iloc[-1],
        'upper_band': upper_band.iloc[-1],
        'lower_band': lower_band.iloc[-1],
        'atr': atr.iloc[-1]
    }

    features = pd.DataFrame([feature_values])
    signal = model.predict(features)[0]

    return {"signal": signal, "symbol": request.stock_symbol}

def run_flask():
    flask_app.run(host='0.0.0.0', port=5000)

def run_fastapi():
    import uvicorn
    uvicorn.run(fastapi_app, host='0.0.0.0', port=8000)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=run_flask)
    fastapi_thread = threading.Thread(target=run_fastapi)

    flask_thread.start()
    fastapi_thread.start()

    flask_thread.join()
    fastapi_thread.join()
