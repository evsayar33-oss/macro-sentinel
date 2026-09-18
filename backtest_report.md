# Macro Sentinel Dynamic — Backtest & Olay Doğrulama Raporu (2018 - 2026)

## 1. Kapsam
Bu rapor, mevcut rejim + bağımsız olay katmanının sentetik tarihsel veri üreticisi üzerindeki davranışını özetler. Metrikler canlı piyasa performansı veya gelecekteki sonuçlar için garanti değildir.

### Mevcut strateji metrikleri
- Yıllıklandırılmış getiri: **%9.17**
- Yıllıklandırılmış volatilite: **%3.54**
- Sharpe: **1.74**
- Maksimum düşüş: **%8.86**
- Calmar: **1.03**
- Toplam getiri: **%119.72**

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
| Macro Sentinel Dynamic | %9.17 | %3.54 | 1.74 | %8.86 | 1.03 | %119.72 |
| Defensive Shield | %8.79 | %3.76 | 1.54 | %14.49 | 0.61 | %112.95 |
| Artemis Dragon | %10.02 | %5.16 | 1.36 | %22.82 | 0.44 | %135.73 |
| Taleb Barbell Asymmetric | %6.35 | %2.18 | 1.54 | %4.66 | 1.36 | %73.85 |

## 4. Araştırma adayları
Bu adaylar sentetik veri üzerinde yalnızca araştırma amacıyla ölçülür; canlı allocation'a otomatik olarak geçirilmez.

| Aday | Yıllık Getiri | Volatilite | Sharpe | Max DD | Calmar |
|---|---:|---:|---:|---:|---:|
| Research: Adaptive Opportunity | %10.95 | %4.32 | 1.84 | %5.64 | 1.94 |
| Research: Adaptive Opportunity + BTC | %13.14 | %4.87 | 2.08 | %6.13 | 2.14 |
| Research: Balanced Trend | %10.99 | %4.41 | 1.81 | %5.93 | 1.85 |
1. Sentetik veri üzerindeki geri çağırma, gerçek tarihsel yeniden oynatma ile aynı şey değildir.
2. Sharpe, Calmar ve drawdown metrikleri tek başına model doğruluğunu kanıtlamaz.
3. Yapısal petrol olayı ayrı bir event katmanı olarak değerlendirilir; named regime ile aynı kavram değildir.
4. Gerçek model kalitesi için ileride walk-forward ve gerçek out-of-sample event outcome audit kullanılmalıdır.
