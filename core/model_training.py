import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_and_save_model():
    # Model dosyasının kaydedileceği dizini belirleme
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Modeli kaydetme yolunu ayarla
    file_path = os.path.join(models_dir, 'random_forest_model.pkl')
    
    # Eğitim verilerini yükleme
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/training_data.csv')
    if not os.path.exists(data_path):
        print(f"Eğitim verisi dosyası bulunamadı: {data_path}")
        return

    data = pd.read_csv(data_path)

    # Özellikler ve etiketleri ayırma
    X = data[['rsi', 'macd', 'stochastic', 'momentum', 'short_sma', 'long_sma']]
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

    print(f'Model başarıyla kaydedildi: {file_path}')

if __name__ == "__main__":
    train_and_save_model()
