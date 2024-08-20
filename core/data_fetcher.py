import yfinance as yf
import pandas as pd
import pickle

def fetch_stock_data(ticker, period='1mo', interval='1d'):
    # Veri çekme
    stock_data = yf.download(ticker, period=period, interval=interval)
    
    # Sadece kapanış fiyatlarını alalım
    close_prices = stock_data['Close']
    return close_prices

def fetch_historical_data(ticker, start_date, end_date):
    # Belirli bir tarih aralığındaki verileri çekme
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

def analyze_stock(ticker):
    data = fetch_stock_data(ticker)

    # Eğitilmiş modeli yükleyelim
    model_path = os.path.join(os.path.dirname(__file__), '../models/random_forest_model.pkl')
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)

    # İndikatör hesaplamaları burada yapılacak
    short_sma = data.rolling(window=10).mean()
    long_sma = data.rolling(window=20).mean()
    rsi = calculate_rsi(data)
    macd, signal_line = calculate_macd(data)
    stochastic = calculate_stochastic(data)
    momentum = calculate_momentum(data)

    # Modelin ihtiyaç duyduğu özellikleri belirleyelim
    features = [
        rsi.iloc[-1],
        macd.iloc[-1],
        stochastic.iloc[-1],
        momentum.iloc[-1],
        short_sma.iloc[-1],
        long_sma.iloc[-1]
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
