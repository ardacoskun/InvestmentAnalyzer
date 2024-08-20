from flask import Flask, request, jsonify
import pandas as pd
import os
import pickle
from core.analysis import (
    calculate_moving_averages, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_stochastic, calculate_atr,
    calculate_momentum, calculate_take_profit_and_stop_loss
)
from core.data_fetcher import analyze_stocks

app = Flask(__name__)

# Eğitilmiş modeli yükleme
model_path = os.path.join(os.path.dirname(__file__), 'models', 'random_forest_model.pkl')

if not os.path.exists(model_path):
    print(f"Model dosyası bulunamadı: {model_path}")
else:
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
        print(f"Model başarıyla yüklendi: {model_path}")

@app.route('/analyze', methods=['POST'])
def analyze():
    tickers = request.json.get("tickers", [])
    
    if not tickers:
        return jsonify({"error": "No tickers provided"}), 400

    # Birden fazla hisseyi tarama ve "buy" sinyali verenleri döndürme
    buy_opportunities = analyze_stocks(tickers)

    return jsonify({
        "buy_opportunities": buy_opportunities
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
