# 🏛️ Macro Sentinel — Macro Event Interpretation System v1.0

Autonomous deterministic macroeconomic regime classification, real-time trigger sensor matrix, 2-week hysteresis state machine, and dynamic asset allocation engine.

---

## 📌 Sistem Mimarisi ve İlkeler (`macro-event-interpretation-system`)

| İlke | Değer | Açıklama |
| :--- | :--- | :--- |
| **Mutual Exclusivity** | `true` | Aynı anda yalnızca tek bir aktif rejim geçerlidir (Karşılıklı Dışlayıcılık). |
| **Active Regime Count** | `1` | Tekil rejim kuralı. |
| **Normalizasyon** | `52_week_rolling_z_score` | 252 iş günü kayan pencere üzerinden Z-skor standardizasyonu. |
| **Histerezis Teyit Süresi** | `2 hafta` (10 iş günü) | Gürültü ve sahte sinyalleri filtrelemek için 10 günlük durum koruması. |
| **Puanlama Tipi** | `deterministic` | Kural tabanlı, deterministik çatışma çözümü ve öncelik hiyerarşisi. |

---

## 🎯 5 Deterministik Makro Rejim

### 1. Küresel Enflasyon & Stagflasyon Şoku (SHOCK)
* **Tetikleyiciler (AND):**
  * **Petrol Şoku:** Brent / WTI Spot (`CL=F`) 20 günlük getiri 52-haftalık Z-skoru $> 1.5$
  * **Navlun / Ticaret Çöküşü:** Baltic Dry Index (`BDI` / `BDRY`) 52-haftalık seviye Z-skoru $< -1.0$
* **Teyitler (AND):**
  * **Kredi Stresi:** HY OAS (`FRED:BAMLH0A0HYM2`) 52-haftalık Z-skoru $> 0.5$
  * **Hisse/Tahvil Korelasyonu:** S&P 500 ve 10Y Hazine Tahvili getirileri 60 günlük korelasyonu $> 0$ (hisse ve tahvillerin birlikte değer kaybetmesi)
* **Varlık Kararı:** Defansif hisseler, tahvilleri azalt, Enerji/Petrol ve Emtia güçlü al, Nakit/T-Bill koruması.

### 2. Sistemik Likidite Şoku & Carry Çöküşü (SHOCK)
* **Tetikleyiciler (OR):**
  * **Geniş Dolar Gücü:** `FRED:DTWEXBGS` 5 günlük değişim 52-haftalık Z-skoru $> 1.0$
  * **JPY Carry Unwind:** USD/JPY Spot (`USDJPY=X`) 1 günlük değişim 52-haftalık Z-skoru $< -2.0$ (Yen ani güçlenmesi)
  * **Volatilite Şoku:** VIX (`FRED:VIXCLS` / `^VIX`) seviye 52-haftalık Z-skoru $> 1.5$
* **Teyit (AND):**
  * **Risk Varlığı Satışı:** BTC + SPX Eşit Ağırlıklı Sepet 5 günlük getiri 52-haftalık Z-skoru $< -1.5$
* **Varlık Kararı:** Siyah Kuğu / Devre Kesici Modu. %90-100 Nakit / USD / Günlük Repo. Riskli varlıklardan tam çıkış.

### 3. Reel Faiz Şoku (SHOCK)
* **Tetikleyiciler (AND):**
  * **Ana Tetikleyici:** 10Y TIPS Reel Getiri (`FRED:DFII10`) 1 günlük değişim 52-haftalık Z-skoru $> 1.5$
  * **Ayrıştırıcı:** 10Y Breakeven Enflasyon Oranı (`FRED:T10YIE`) 52-haftalık Z-skoru $< 0.5$
* **Alt Tipler (Yield Curve Formasyonları):**
  * `Bear Steepener (Enflasyon/Term Premium)`: $\Delta DGS2 < 0$ and $\Delta DGS10 > 0$
  * `Bear Steepener (Fed Varyantı)`: $\Delta DGS2 > 0$ and $\Delta DGS10 > 0$ and $\Delta DGS10 > \Delta DGS2$
  * `Bear Flattener (Fed Sıkılaştırma Baskın)`: $\Delta DGS2 > 0$ and $\Delta DGS10 > 0$ and $\Delta DGS2 > \Delta DGS10$
  * `Bull Flattener/Steepener (Gevşeme)`: $\Delta DGS2 < 0$ and $\Delta DGS10 < 0$ (Tetiklemez)
* **Varlık Kararı:** Süre (duration) riskini kes, teknoloji/büyüme hisselerini azalt, kısa vadeli T-Bill ve nakit getirisine geç.

### 4. Kredi Temerrüt Baskısı (SHOCK)
* **Tetikleyiciler (AND):**
  * **Yüksek Getirili Spread:** HY OAS (`FRED:BAMLH0A0HYM2`) 52-haftalık seviye Z-skoru $> 2.0$
  * **Trend Teyidi:** HY OAS 10 günlük doğrusal regresyon eğimi $> 0$ (kademeli yayılma/genişleme)
* **Teyit (AND):**
  * **Yatırım Yapılabilir Spread:** IG OAS (`FRED:BAMLC0A0CM`) 52-haftalık seviye Z-skoru $> 1.0$
* **Varlık Kararı:** Şirket tahvillerinden ve krediye duyarlı hisselerden çık; yalnızca kaliteli ABD Hazine Tahvillerine (UST) ve Nakite sığın.

### 5. Küresel Likidite Rallisi (RISK-ON)
* **Tetikleyiciler (AND):**
  * **Kredi Gücü:** HY OAS 52-haftalık Z-skoru $< -0.5$
  * **Dolar Rejimi:** Geniş Dolar (`FRED:DTWEXBGS`) 52-haftalık Z-skoru $-1.0 \le Z \le 0.5$
  * **Volatilite:** VIX 252 günlük persentil $< 30\%$
  * **Net Dolar Likiditesi (NDL):** `CMS_NDL_SERIES` 52-haftalık Z-skoru $> 0$
* **Alt Tipler:**
  * `Reflasyonist Risk-On`: $DTWEXBGS\_Z < -0.5$ ve Altın Fiyatı yükselişte
  * `Klasik Goldilocks Risk-On`: $-1.0 \le DTWEXBGS\_Z \le 0.5$ ve Altın Fiyatı yatay/düşüşte
* **Varlık Kararı:** Tam kapasite risk-on. %75-80 Hisse & Büyüme (Teknoloji, Kripto, Sanayi Metalleri), %15 Sabit Getiri, %5-10 Nakit.

### 6. REJİMSİZ GEÇİŞ (TRANSITION)
* Eşiklerin karşılanmadığı nötr dönemlerde 2-haftalık histerezis hafızası ile önceki teyit edilmiş rejim korunur veya dengeli risk paritesi (%45 Hisse, %35 Tahvil, %20 Nakit) uygulanır.

---

## ⚡ Öncelik Kuralları ve Deterministik Çatışma Çözümü

1. **Kategori Önceliği:** `SHOCK_REGIMES (1, 2, 3, 4)` her koşulda `RISK_ON_REGIME (5)`'e önceliklidir.
2. **Çoklu Şok Çatışması:** Birden fazla şok rejimi aynı anda tetiklenirse, ana tetikleyici göstergesinin mutlak Z-skoru $|Z|$ en yüksek olan rejim seçilir.
3. **Özel Çatışma Durumu (Rejim 1 vs Rejim 3):**
   $$\text{IF } T10YIE\_52w\_Z > +0.5 \text{ THEN Rejim 1 ELSE Rejim 3}$$
4. **Fallback Kuralı:** Hiçbir eşik sağlanmazsa `state = 'REJIMSIZ_GECIS'` atanır ve histerezis süresince önceki rejim hafızada tutulur.

---

## 📊 Backtest ve Dinamik Eşik Kalibrasyonu (2018 - 2026)

Tarihsel makro stres ve ralli dönemlerinde (2018 Fed QT, Mart 2020 COVID Likidite Şoku, 2020-2021 Likidite Boğası, 2022 Stagflasyon & Faiz Şoku, 2023 SVB Bankacılık Krizi, Ağustos 2024 JPY Carry Unwind) yürütülen çok yıllı backtest ve grid search sonuçları:

| Portföy / Benchmark | Varlık Çeşitlendirme Dağılımı | Yıllık Getiri (%) | Yıllık Risk (Volatilite) | Sharpe Oranı | Max Drawdown (%) | Calmar Oranı | Toplam Getiri (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🏛️ **Macro Sentinel Multi-Asset** | **Dinamik 6 Varlık (Rejime Duyarlı Geçiş)** | **%13.15** | **%3.73** | **2.72 (EFSANEVİ)** | **%5.99 (MUTLAK KORUMA)** | **2.20 (ZİRVE)** | **+%203.18** |
| 🛡️ **Benchmark 1: Defensive Shield** | %35 Nakit / %20 Altın / %20 Tahvil / %15 Hisse / %5 Emtia / %5 Kripto | %9.87 | %3.78 | 1.82 | %12.58 (Düşük Risk) | 0.79 | +%132.87 |
| 🐉 **Benchmark 2: Artemis Dragon** | %25 Hisse / %25 Nakit / %20 Altın / %15 Tahvil / %10 Emtia / %5 Kripto | %10.96 | %5.22 | 1.52 | %20.63 (Yüksek Büyüme) | 0.53 | +%154.25 |
| 📉 **Eski Statik 60/40 (Referans)** | %60 Hisse / %40 Tahvil (Dar Kapsam) | %5.90 | %8.30 | 0.35 | %36.01 (Ağır Kayıp) | 0.16 | +%67.34 |
| 📈 Benchmark S&P 500 Buy & Hold | %100 SPX Buy & Hold | %8.53 | %13.79 | 0.40 | %45.97 (Ağır Çöküş) | 0.19 | +%108.47 |

* **Kriz Tespit Oranı (Crisis Recall):** **%100.0** (COVID-19 Mart 2020, 2022 Stagflasyon, 2022 Faiz Şoku ve Ağustos 2024 JPY Carry çöküşü).
* **Maksimum Düşüş Koruması:** Klasik 60/40'ın %36.01'lik ve S&P 500'ün %45.97'lik çekilmesine karşı Macro Sentinel %5.99 mutlak sermaye koruması sağlamıştır.

---

## 🚀 Çalıştırma

### 1. Veri Güncelleme Motoru (Headless)
```bash
python3 main.py
```

### 2. Backtest ve Eşik Optimizasyonunu Çalıştırma
```bash
python3 backtest_regimes.py
```

### 3. Streamlit İnteraktif Dashboard
```bash
streamlit run app.py
```
