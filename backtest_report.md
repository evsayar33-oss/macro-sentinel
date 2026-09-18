# Macro Sentinel Dynamic — Backtest & Olay Doğrulama Raporu (2018 - 2026)

## 1. Kapsam
Bu rapor, mevcut rejim + bağımsız olay katmanının sentetik tarihsel veri üreticisi üzerindeki davranışını özetler. Metrikler canlı piyasa performansı veya gelecekteki sonuçlar için garanti değildir.

### Mevcut strateji metrikleri
- Yıllıklandırılmış getiri: **%16.07**
- Yıllıklandırılmış volatilite: **%5.25**
- Sharpe: **2.49**
- Maksimum düşüş: **%7.28**
- Calmar: **2.21**
- Toplam getiri: **%280.90**

## 2. Rejim ve olay geri çağırma kontrolleri
- 2020 likidite şoku: **DETECTED**
- 2022 stagflasyon: **DETECTED**
- 2022 reel faiz şoku: **DETECTED**
- 2024 JPY carry olayı: **DETECTED**
- 2022 yapısal petrol olayı: **DETECTED**

## 3. Benchmarklar
Benchmarklar yalnızca karşılaştırmalı bağlam sağlar; sonuçlar veri üretim sürecine ve test varsayımlarına bağlıdır.

| Strateji | Yıllık Getiri | Volatilite | Sharpe | Max DD | Calmar | Toplam Getiri |
|---|---:|---:|---:|---:|---:|---:|
| Macro Sentinel Dynamic | %16.07 | %5.25 | 2.49 | %7.28 | 2.21 | %280.90 |
| Defensive Shield | %8.79 | %3.76 | 1.54 | %14.49 | 0.61 | %112.95 |
| Artemis Dragon | %10.02 | %5.16 | 1.36 | %22.82 | 0.44 | %135.73 |
| Taleb Barbell Asymmetric | %6.35 | %2.18 | 1.54 | %4.66 | 1.36 | %73.85 |

## 4. Araştırma adayları
Bu adaylar sentetik veri üzerinde yalnızca araştırma amacıyla ölçülür; canlı allocation'a otomatik olarak geçirilmez.

| Aday | Yıllık Getiri | Volatilite | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|
1. Sentetik veri üzerindeki geri çağırma, gerçek tarihsel yeniden oynatma ile aynı şey değildir.
2. Sharpe, Calmar ve drawdown metrikleri tek başına model doğruluğunu kanıtlamaz.
3. Yapısal petrol olayı ayrı bir event katmanı olarak değerlendirilir; named regime ile aynı kavram değildir.
4. Gerçek model kalitesi için ileride walk-forward ve gerçek out-of-sample event outcome audit kullanılmalıdır.

## 5. Çoklu-seed üretim stratejisi kontrolü (sentetik)
Aynı üretim allocation mimarisi beş farklı sentetik seed üzerinde ayrı ayrı çalıştırıldı. Bu tablo canlı piyasa performansı değildir; yalnızca parametrelerin tek bir sentetik senaryoya aşırı bağımlı olup olmadığını kontrol etmek içindir.

| Seed | Yıllık Getiri | Max DD | Calmar | Sharpe | Toplam Getiri |
|---:|---:|---:|---:|---:|---:|
| 7 | %18.56 | %5.63 | 3.30 | 2.75 | %361.07 |
| 19 | %20.24 | %5.46 | 3.71 | 3.01 | %423.23 |
| 42 | %16.07 | %7.28 | 2.21 | 2.49 | %280.90 |
| 71 | %21.45 | %6.13 | 3.50 | 3.31 | %472.35 |
| 101 | %17.91 | %6.64 | 2.70 | 2.69 | %338.87 |
| **Medyan** | **%18.56** | **%6.13** | **3.30** | **2.75** | **%361.07** |

## 6. Strateji mimarisi
Üretim allocation'ı artık yalnızca sabit rejim yüzdelerinden oluşmuyor. Rejim bazlı risk bütçesi, SPX 20/100 günlük trendi, VIX persentili ve BTC 60 günlük trendi ile sınırlı biçimde ayarlanıyor. Sistemik likidite şokunda risk bütçesi sert biçimde düşürülüyor. Petrol olayı ise bu risk bütçesinin üzerine yalnızca doğrulanmış event olarak uygulanabiliyor.

Önemli: Bu testler gerçek fiyat serisi üzerinde değildir. Sonuçlar canlı getiri beklentisi veya garanti olarak kullanılmamalıdır. Gerçek veri walk-forward doğrulaması yapılmadan parametrelerin daha ileri otomatik değiştirilmesi kapalıdır.

## 5. Çoklu-seed üretim stratejisi kontrolü (sentetik)
Aynı üretim allocation mimarisi beş farklı sentetik seed üzerinde ayrı ayrı çalıştırıldı.

| Seed | Yıllık Getiri | Max DD | Calmar | Sharpe | Toplam Getiri |
|---:|---:|---:|---:|---:|---:|
| 7 | %18.56 | %5.63 | 3.30 | 2.75 | %361.07 |
| 19 | %20.24 | %5.46 | 3.71 | 3.01 | %423.23 |
| 42 | %16.07 | %7.28 | 2.21 | 2.49 | %280.90 |
| 71 | %21.45 | %6.13 | 3.50 | 3.31 | %472.35 |
| 101 | %17.91 | %6.64 | 2.70 | 2.69 | %338.87 |
| **Medyan** | **%18.56** | **%6.13** | **3.30** | **2.75** | **%361.07** |

## 6. Araştırma stratejileri
Beş seed üzerindeki medyan sonuçlar:

| Strateji | Medyan Yıllık Getiri | Medyan Max DD | Medyan Calmar | Medyan Sharpe | Medyan Toplam Getiri |
|---|---:|---:|---:|---:|---:|
| Macro Sentinel Dynamic (Production) | %18.56 | %6.13 | 3.30 | 2.75 | %361.07 |
| Adaptive Opportunity + BTC | %16.05 | %4.89 | 3.21 | 2.53 | %280.37 |
| Balanced Trend | %11.94 | %4.87 | 2.37 | 1.94 | %175.32 |
| Adaptive Opportunity | %11.77 | %4.52 | 2.44 | 1.96 | %171.49 |

## 7. Canlıya alma sınırı
Production allocation artık bounded Adaptive Opportunity + BTC mimarisini kullanır; ancak parametreler günlük gerçekleşen getirilerden otomatik olarak değiştirilmez. Sentetik sonuçlar yalnızca mimari araştırma içindir. Gerçek piyasa verisi ile ileriye dönük out-of-sample/walk-forward doğrulama tamamlanmadan yeni parametre mutasyonu yapılmaz.


## 8. Yorumlama notları
1. Sentetik veri üzerindeki geri çağırma, gerçek tarihsel yeniden oynatma ile aynı şey değildir.
2. Getiri, Sharpe, Calmar ve drawdown metrikleri tek başına model doğruluğunu kanıtlamaz.
3. Yapısal petrol olayı ayrı bir event katmanı olarak değerlendirilir; named regime ile aynı kavram değildir.
4. Canlı model kalitesi için gerçek piyasa verisi üzerinde forward-return outcome audit, işlem maliyeti ve walk-forward doğrulaması gerekir.
