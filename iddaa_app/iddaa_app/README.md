# İddaa Tahmin Modeli — Kurulum

## 1 — Gereksinimler
Python 3.9 veya üzeri

## 2 — Kurulum (tek seferlik)

```bash
# Terminali aç, bu klasörün içine gel
cd iddaa_app

# Sanal ortam oluştur (tavsiye edilir)
python -m venv venv

# Aktif et
# Mac / Linux:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Kütüphaneleri yükle
pip install -r requirements.txt
```

## 3 — Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcı otomatik açılır → http://localhost:8501

## 4 — Her Hafta Kullanım

1. Uygulamayı başlat: `streamlit run app.py`
2. Sol panelden yeni `.txt` dosyasını yükle
3. Model ayarlarını istersen değiştir
4. Tahminler ve dashboard otomatik gelir

## 5 — Klasör Yapısı

```
iddaa_app/
├── app.py            # Ana uygulama
├── requirements.txt  # Kütüphaneler
└── README.md         # Bu dosya
```

## 6 — Notlar

- Her dosya yüklemesinde model sıfırdan eğitilir
- Oynanan maçlar eğitim için, oynanmayanlar tahmin için kullanılır
- Güven eşiği sol panelden ayarlanabilir
- Tahminler CSV olarak indirilebilir
