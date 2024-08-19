import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_and_save_model():
    # Eğitim verilerini yükleme
    data = pd.read_csv('../data/training_data.csv')

    # Özellikler ve etiketleri ayırma
    X = data[['rsi', 'macd', 'stochastic', 'momentum', 'short_sma', 'long_sma']]
    y = data['label']

    # Veriyi eğitim ve test olarak ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest modelini eğitme
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Test verisi üzerinde tahmin yapma
    y_pred = model.predict(X_test)

    # Modelin doğruluk oranını kontrol etme
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Eğitilen modeli dosyaya kaydetme
    with open('../models/random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

if __name__ == "__main__":
    train_and_save_model()
