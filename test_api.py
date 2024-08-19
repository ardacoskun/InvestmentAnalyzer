import requests
import json

# API endpoint'ini belirleyelim
url = "http://127.0.0.1:5000/analyze"

# Test etmek için fiyat verileri
data = {
    "prices": [100, 101, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85],
    "high_prices": [101, 102, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86],
    "low_prices": [99, 100, 101, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84]
}


# POST isteği gönderiyoruz
response = requests.post(url, json=data)

# Gelen yanıtı ekrana yazdırıyoruz
print("Status Code:", response.status_code)
print("Response JSON:", response.json())
