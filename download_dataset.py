import pandas as pd

# URL
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-total-female-births.csv"

# Veriyi oku
df = pd.read_csv(url)

# İstersen dosyaya kaydedebilirsin
# df.to_csv("daily-min-temperatures.csv", index=False)
df.to_csv("daily-total-female-births.csv", index=False)

print("Veri başarıyla indirildi ve kaydedildi!")