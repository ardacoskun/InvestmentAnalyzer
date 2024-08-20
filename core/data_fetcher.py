import yfinance as yf
import pandas as pd
import pickle

def fetch_stock_data(ticker, period='1mo', interval='1d'):
    # Veri çekme
    stock_data = yf.download(ticker, period=period, interval=interval)
    
    # Sadece kapanış fiyatlarını alalım
    close_prices = stock_data['Close']
    return close_prices

def analyze_stock(ticker):
    data = fetch_stock_data(ticker)

    # Eğitilmiş modeli yükleyelim
    with open('../models/random_forest_model.pkl', 'rb') as model_file:
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

# İndikatör hesaplama fonksiyonlarını buraya ekleyebilirsiniz.
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Diğer indikatör fonksiyonları da buraya eklenebilir...

if __name__ == "__main__":
    tickers = ["ASELS.IS", "GARAN.IS", "THYAO.IS", "SASA.IS", "BIMAS.IS"]  # Örnek hisse listesi
    buy_opportunities = analyze_stocks(tickers)
    print("Alım fırsatı sunan hisseler:", buy_opportunities)
