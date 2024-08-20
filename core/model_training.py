import sys
import os

# Mevcut dosyanın dizinini alın ve proje kök dizinine ekleyin
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
from core.data_fetcher import fetch_stock_data
from core.analysis import calculate_rsi, calculate_macd, calculate_stochastic, calculate_momentum, calculate_moving_averages, calculate_bollinger_bands, calculate_atr

def prepare_training_data():
    tickers = ["ASELS.IS", "GARAN.IS", "THYAO.IS", "SASA.IS", "BIMAS.IS"]
    start_date = "2018-01-01"
    end_date = "2023-01-01"

    all_data = []

    for ticker in tickers:
        data = fetch_stock_data(ticker, start_date=start_date, end_date=end_date)

        if data is None:
            continue
        
        # İndikatör hesaplamalarını yapalım
        short_sma, long_sma = calculate_moving_averages(data['Close'])
        rsi = calculate_rsi(data['Close'])
        macd, signal_line = calculate_macd(data['Close'])
        stochastic = calculate_stochastic(data['Close'])
        momentum = calculate_momentum(data['Close'])
        upper_band, lower_band = calculate_bollinger_bands(data['Close'])
        atr = calculate_atr(data['Close'], data['High'], data['Low'])

        # Hesaplanan indikatörleri dataframe'e ekleyelim
        data['rsi'] = rsi
        data['macd'] = macd
        data['stochastic'] = stochastic
        data['momentum'] = momentum
        data['short_sma'] = short_sma
        data['long_sma'] = long_sma
        data['upper_band'] = upper_band
        data['lower_band'] = lower_band
        data['atr'] = atr

        # Daha dinamik bir etiketleme
        data['label'] = data.apply(lambda row: 'buy' if row['rsi'] < 40 and row['macd'] > 0 and row['short_sma'] > row['long_sma'] else 'sell' if row['rsi'] > 60 and row['macd'] < 0 else 'hold', axis=1)

        # DataFrame'i birleştirelim
        all_data.append(data[['rsi', 'macd', 'stochastic', 'momentum', 'short_sma', 'long_sma', 'upper_band', 'lower_band', 'atr', 'label']])

    # Tüm verileri tek bir DataFrame'de birleştirelim
    final_data = pd.concat(all_data)

    # CSV dosyasına kaydedelim
    data_dir = os.path.join(os.path.dirname(__file__), '../data/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    final_data.to_csv(os.path.join(data_dir, 'training_data.csv'), index=False)
    print(f"Eğitim verisi başarıyla kaydedildi: {os.path.join(data_dir, 'training_data.csv')}")

def train_and_save_model():
    # Klasörün var olup olmadığını kontrol et ve yoksa oluştur
    models_dir = os.path.join(os.path.dirname(__file__), '../models/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Modeli kaydetme yolunu ayarla
    file_path = os.path.join(models_dir, 'random_forest_model.pkl')
    
    # Eğitim verilerini yükleme
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), '../data/training_data.csv'))

    # Özellikler ve etiketleri ayırma
    X = data[['rsi', 'macd', 'stochastic', 'momentum', 'short_sma', 'long_sma', 'upper_band', 'lower_band', 'atr']]
    y = data['label']

    # Veriyi eğitim ve test olarak ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest modelini eğitme
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Modelin doğruluk oranını kontrol etme
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Eğitilen modeli dosyaya kaydetme
    with open(file_path, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"Model başarıyla kaydedildi: {file_path}")

if __name__ == "__main__":
    prepare_training_data()  # Gerçek verileri çekme işlemi
    train_and_save_model()   # Modeli eğitme ve kaydetme
