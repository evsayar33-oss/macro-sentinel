# Macro Sentinel v3.0 — Adaptive Cross-Asset Strategy Layer

Bu paket üç parçadan oluşur:

- `adaptive_strategy_layer.py`: üretim strateji katmanı.
- `strategy_research.py`: gerçek geçmiş yeterliyse walk-forward / Pareto araştırma motoru.
- `main_integration.patch`: mevcut `main.py` içine entegrasyon değişikliği.
- `strategy_config_patch.json`: `regime_config.json` içindeki `research_strategies.adaptive_layer` altına eklenecek ayarlar.

## Temel davranış

Normal koşullarda sermaye, risk-ayarlı fırsat skorlarına göre beş risk varlığına dağıtılır ve nakit kalan risk bütçesini temsil eder.

Aynı anda birçok risk varlığı düşüyor, aralarındaki korelasyon yükseliyor ve VIX/likidite koşulları bozuluyorsa `CAPITAL_PRESERVATION_*` modu devreye girer ve risk bütçesi sert biçimde azaltılır.

Hard stress durumunda toplam risk bütçesi varsayılan olarak `%10` ile sınırlandırılır. Sistem gold'u otomatik olarak "güvenli" kabul etmez; kriz günlerinde nakit gerçek savunma aracıdır.

Olay katmanındaki doğrulanmış petrol olayı toplam riski artırmak zorunda değildir; mevcut risk bütçesi içinde emtiaya öncelik verir.

## Önemli

Araştırma motoru tek bir backtest sonucunu "en iyi strateji" olarak ilan etmez. Pareto önünde yer alan adayları çıkarır. Üretime otomatik terfi yoktur. Gerçek veride yeterli tarih ve out-of-sample walk-forward şarttır.
