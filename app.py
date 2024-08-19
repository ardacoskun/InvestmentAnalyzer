from flask import Flask, request, jsonify
import pandas as pd
import pickle
from core.analysis import (
    calculate_moving_averages, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_stochastic, calculate_atr,
    calculate_momentum, calculate_take_profit_and_stop_loss
)

app = Flask(__name__)

# Eğitilmiş modeli yükleme
with open('models/random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    prices = pd.Series(data['prices'])  # Fiyat verilerini alıyoruz
    high_prices = pd.Series(data.get('high_prices', prices))  # Opsiyonel olarak yüksek fiyatlar
    low_prices = pd.Series(data.get('low_prices', prices))  # Opsiyonel olarak düşük fiyatlar

    # Teknik analizleri hesaplıyoruz
    short_sma, long_sma = calculate_moving_averages(prices)
    rsi = calculate_rsi(prices)
    macd, signal_line = calculate_macd(prices)
    upper_band, lower_band = calculate_bollinger_bands(prices)
    stochastic = calculate_stochastic(prices)
    atr = calculate_atr(prices, high_prices, low_prices)
    momentum = calculate_momentum(prices)
    last_price = prices.iloc[-1]
    take_profit, stop_loss = calculate_take_profit_and_stop_loss(last_price)

    # Modeli kullanarak sinyal üretme
    features = [rsi.iloc[-1], macd.iloc[-1], stochastic.iloc[-1], momentum.iloc[-1], short_sma.iloc[-1], long_sma.iloc[-1]]
    signal = model.predict([features])[0]  # Modelden tahmin edilen sinyal

    return jsonify({
        "signal": signal,
        "confidence": 0.99,
        "technical_indicators": {
            "short_sma": short_sma.iloc[-1] if not pd.isna(short_sma.iloc[-1]) else None,
            "long_sma": long_sma.iloc[-1] if not pd.isna(long_sma.iloc[-1]) else None,
            "rsi": rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else None,
            "macd": macd.iloc[-1],
            "signal_line": signal_line.iloc[-1],
            "upper_band": upper_band.iloc[-1],
            "lower_band": lower_band.iloc[-1],
            "stochastic": stochastic.iloc[-1],
            "atr": atr.iloc[-1],
            "momentum": momentum.iloc[-1]
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
