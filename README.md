Modüler bir kredi risk değerlendirme web uygulaması: XGBoost modeli temerrüt olasılığını tahmin eder,
kural motoru açıklanabilir iyileştirme önerileri üretir ve Streamlit paneli sonucu Plotly gösterge
grafiğiyle sunar.

## Özellikler
- **ML hattı:** `scikit-learn` `Pipeline` + `ColumnTransformer` (ölçekleme, one-hot kodlama, türetilmiş oran özellikleri) ve XGBoost.
- **Dengesiz veri:** Katmanlı (stratified) eğitim/test ayırma ve `scale_pos_weight`; ROC-AUC ve sınıflandırma raporu ile değerlendirme.
- **Açıklanabilirlik:** Borç/gelir oranı, kredi/gelir oranı, çalışma süresi, birikim gibi etkenleri kapsayan kural tabanlı öneriler (`src/recommendations.py`).
- **Panel:** Yan menüden girdi alma, onay ihtimali göstergesi (%0-100) ve renk kodlu risk grubu (yeşil / sarı / kırmızı).

## Proje Yapısı
```
credit-risk-engine/
├── data/german_credit_data.csv        # otomatik üretilir (sentetik) veya kendi verinizi koyun
├── models/
│   ├── train_model.py
│   └── credit_model.pkl               # eğitimle otomatik üretilir
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   └── recommendations.py
├── app.py
├── requirements.txt
└── README.md
```

## Hızlı Başlangıç
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python models/train_model.py
streamlit run app.py
```

> Uygulamayı açmadan önce eğitim betiğini çalıştırmak zorunludur; model dosyası (`credit_model.pkl`) bu betikle oluşur.

## Veri Notu
Herkese açık German Credit veri setinde gelir, borç/gelir oranı ve temerrüt etiketi bulunmaz.
`data/german_credit_data.csv` dosyası beklenen şemaya uymuyorsa
(`age, monthly_income, loan_amount, loan_term_months, debt_to_income, employment_years, housing, savings_level, default`)
ya da hiç yoksa, eğitim betiği bu şemada **sentetik** ve dengesiz (yaklaşık %15-20 temerrüt) bir veri seti üretir.

**Önemli:** Sentetik veri, kodda tanımlı bir olasılık formülünden üretilir. Model bu formülü öğrenir; dolayısıyla
raporlanan AUC değeri gerçek dünya performansını yansıtmaz. Sistemi demo dışında kullanmak için kendi gerçek
veri setinizi aynı şemada `data/` klasörüne koyun.

## Tasarım İlkeleri
- **Tek sorumluluk (SRP):** Ön işleme, eğitim, kurallar ve arayüz ayrı modüllerde yer alır.
- **Açık/kapalı (OCP):** Yeni bir tavsiye eklemek için `RULES` listesine bir `Rule` eklemek yeterlidir; başka kodu değiştirmek gerekmez.
- **Tekrarlanabilirlik:** Sabit rastgelelik tohumları kullanılır ve tüm ön işleme adımları model ile birlikte serileştirilir.

## Yol Haritası
SHAP tabanlı özellik katkıları, olasılık kalibrasyonu, Docker imajı, FastAPI ile skorlama servisi.

## Sorumluluk Reddi
Yalnızca eğitim amaçlıdır. Finansal tavsiye niteliği taşımaz ve gerçek kredi kararlarında kullanılamaz.
