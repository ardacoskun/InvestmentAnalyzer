import pandas as pd

def calculate_moving_averages(prices, short_window=10, long_window=20):
    short_sma = prices.rolling(window=short_window).mean()
    long_sma = prices.rolling(window=long_window).mean()
    return short_sma, long_sma

def calculate_rsi(prices, window=14):
    delta = prices.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices):
    short_ema = prices.ewm(span=12, adjust=False).mean()
    long_ema = prices.ewm(span=26, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=9, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(prices, window=20):
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return upper_band, lower_band

def calculate_stochastic(prices, window=14):
    lowest_low = prices.rolling(window=window).min()
    highest_high = prices.rolling(window=window).max()
    stochastic = 100 * (prices - lowest_low) / (highest_high - lowest_low)
    return stochastic

def calculate_atr(prices, high, low, window=14):
    high_low = high - low
    high_close = (high - prices.shift()).abs()
    low_close = (low - prices.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_momentum(prices, window=10):
    momentum = prices.diff(window)
    return momentum

def calculate_take_profit_and_stop_loss(last_price, take_profit_percentage=5, stop_loss_percentage=3):
    take_profit = last_price * (1 + take_profit_percentage / 100)
    stop_loss = last_price * (1 - stop_loss_percentage / 100)
    return take_profit, stop_loss
