# Chatbot-Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)

## Proje Açıklaması
Bu proje, üniversite öğrencilerinin sıkça sorduğu soruları yanıtlamak ve genel bilgilendirme yapmak amacıyla geliştirilmiş bir chatbot uygulamasıdır. Chatbot, kullanıcıların metin tabanlı mesajlarına otomatik yanıtlar vererek onların sorularını cevaplar.

## Özellikler
- **Doğal Dil İşleme (NLP):** Kullanıcıların metin tabanlı sorularını anlamak için NLP teknikleri kullanılır.
- **Eğitilebilir Model:** Yeni verilerle eğitilerek geliştirilebilir.
- **Çeşitli Soru Türleri:** Selamlama, veda, üniversite hakkında bilgi ve teşekkür gibi çeşitli soru türlerine yanıt verebilir.
- **Hata Yönetimi:** Anlamsız veya tanınmayan sorulara uygun geri bildirimde bulunur.

## Kullanılan Teknolojiler
- **Python:** Uygulamanın geliştirilmesi için kullanılan ana programlama dili.
- **TensorFlow/Keras:** Makine öğrenimi modeli ve doğal dil işleme (NLP) için kullanılır.
- **Flask:** Web uygulamasını çalıştırmak için kullanılan mikro web çatısı.
- **NLTK (Natural Language Toolkit):** Doğal dil işleme için kullanılan Python kütüphanesi.
- **Matplotlib:** Model eğitim sürecini görselleştirmek için kullanılan grafik kütüphanesi.

## Kurulum ve Çalıştırma
### Gereksinimler
- Python 3.8 veya üzeri
- Pip (Python paket yöneticisi)

### Adım 1: Depoyu Klonlayın
```bash
git clone https://github.com/mregodd/Chatbot-Project.git
cd Chatbot-Project
````

### Adım 2: Sanal Ortam Oluşturun ve Aktifleştirin
```python -m venv chatbot-env
source chatbot-env/bin/activate  # Windows için: chatbot-env\Scripts\activate
```

### Adım 3: Gerekli Paketleri Yükleyin
```pip install -r requirements.txt```

### Adım 4: Modeli Eğitin
```python models/model_train.py```

### Adım 5: Uygulamayı Başlatın
```python app.py```

### Adım 6: Tarayıcıda Uygulamayı Açın
Tarayıcınızda http://127.0.0.1:5000 adresine gidin ve chatbot'u kullanmaya başlayın.

# Dosya Yapısı

- **data/**: Chatbot'un öğrenme ve yanıt verme verilerini içerir.
  - intents.json

- **models/**: Makine öğrenimi modelleri ve eğitim dosyalarını içerir.
  - chatbot_model.py
  - model_train.py

- **static/**: Statik dosyalar (CSS, JavaScript, görüntüler vb.).
  - style.css

- **templates/**: HTML şablon dosyaları.
  - index.html

- **app.py**: Flask uygulamasının ana dosyası.
- **requirements.txt**: Gerekli Python paketlerini listeler.
- **README.md**: Proje hakkında bilgi verir.

## Katkıda Bulunma
Katkıda bulunmak için lütfen bir pull request oluşturun veya bir issue açın. Her türlü katkı ve geri bildirim değerlidir.

## İletişim
Herhangi bir soru veya geri bildirim için mregodd ile iletişime geçebilirsiniz.
