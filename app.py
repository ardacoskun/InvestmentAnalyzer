from flask import Flask, request, jsonify
import pandas as pd
import pickle
from core.data_fetcher import fetch_stock_data
from core.analysis import (
    calculate_moving_averages, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_stochastic, calculate_atr,
    calculate_momentum, calculate_buy_or_sell_signal
)

app = Flask(__name__)

# Eğitilmiş modeli yükleme
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    tickers = data.get('tickers', [])  # "tickers" key'ini alıyoruz
    
    buy_opportunities = []
    sell_opportunities = []

    for ticker in tickers:
        # Hisse senedi verisini çekiyoruz
        stock_data = fetch_stock_data(ticker)

        if stock_data is None:
            continue

        # Teknik analizleri hesaplıyoruz
        short_sma, long_sma = calculate_moving_averages(stock_data['Close'])
        rsi = calculate_rsi(stock_data['Close'])
        macd, signal_line = calculate_macd(stock_data['Close'])
        upper_band, lower_band = calculate_bollinger_bands(stock_data['Close'])
        stochastic = calculate_stochastic(stock_data['Close'])
        atr = calculate_atr(stock_data['Close'], stock_data['High'], stock_data['Low'])
        momentum = calculate_momentum(stock_data['Close'])

        # İndikatör değerlerini ekrana yazdırarak kontrol edelim
        print(f"Ticker: {ticker}, RSI: {rsi.iloc[-1]}, MACD: {macd.iloc[-1]}, Stochastic: {stochastic.iloc[-1]}, Momentum: {momentum.iloc[-1]}, Short SMA: {short_sma.iloc[-1]}, Long SMA: {long_sma.iloc[-1]}, Upper Band: {upper_band.iloc[-1]}, Lower Band: {lower_band.iloc[-1]}, ATR: {atr.iloc[-1]}")

        # Modeli kullanarak sinyal üretme
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

        # Özellik adlarını belirleyerek tahmin yapma
        features = pd.DataFrame([feature_values])
        signal = model.predict(features)[0]  # Modelden tahmin edilen sinyal

        # Model tahminini ekrana yazdıralım
        print(f"Model Tahmini: {signal}")

        # Al veya sat sinyalini belirliyoruz
        if signal == "buy":
            buy_opportunities.append(ticker)
        elif signal == "sell":
            sell_opportunities.append(ticker)

    return jsonify({
        "buy_opportunities": buy_opportunities,
        "sell_opportunities": sell_opportunities
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
