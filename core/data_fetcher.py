import yfinance as yf
import pandas as pd
import pickle
import time

from core.analysis import calculate_atr, calculate_bollinger_bands, calculate_macd, calculate_momentum, calculate_moving_averages, calculate_rsi, calculate_stochastic

def fetch_stock_data(ticker, start_date=None, end_date=None, period='1mo', interval='1d'):
    try:
        if start_date and end_date:
            stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        else:
            stock_data = yf.download(ticker, period=period, interval=interval)
        return stock_data
    except Exception as e:
        print(f"Veri çekilirken hata oluştu: {e}")
        return None

def analyze_stock(ticker):
    data = fetch_stock_data(ticker)

    # Eğitilmiş modeli yükleyelim
    with open('../models/random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    # İndikatör hesaplamaları burada yapılacak
    short_sma, long_sma = calculate_moving_averages(data['Close'])
    rsi = calculate_rsi(data['Close'])
    macd, signal_line = calculate_macd(data['Close'])
    stochastic = calculate_stochastic(data['Close'])
    momentum = calculate_momentum(data['Close'])
    upper_band, lower_band = calculate_bollinger_bands(data['Close'])
    atr = calculate_atr(data['Close'], data['High'], data['Low'])

    # Modelin ihtiyaç duyduğu özellikleri belirleyelim
    features = [
        rsi.iloc[-1],
        macd.iloc[-1],
        stochastic.iloc[-1],
        momentum.iloc[-1],
        short_sma.iloc[-1],
        long_sma.iloc[-1],
        upper_band.iloc[-1],
        lower_band.iloc[-1],
        atr.iloc[-1]
    ]

    # Modelden tahmin edilen sinyali alalım
    signal = model.predict([features])[0]

    return signal

def analyze_stocks(tickers):
    buy_signals = []

    for ticker in tickers:
        try:
            signal = analyze_stock(ticker)
            if signal == "buy":
                buy_signals.append(ticker)
        except Exception as e:
            print(f"Veri çekilirken hata oluştu: {ticker} - {e}")

    return buy_signals
