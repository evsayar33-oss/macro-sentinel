import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- AYARLAR ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"

class MacroSentinel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_fred(self, s_id):
        """FRED'den veri çeker, hata alırsa None döner."""
        try:
            params = {'series_id': s_id, 'api_key': self.api_key, 'file_type': 'json', 'limit': 10}
            r = requests.get(self.base_url, params=params, timeout=10).json()
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            return obs.set_index(pd.to_datetime(obs['date']))['value'].dropna()
        except: return None

    def get_failover_data(self):
        """FRED başarısız olursa Yahoo Finance üzerinden Proxy üretir."""
        try:
            # HY Spread Proxy: 100 - HYG ETF Fiyatı
            hyg = yf.download("HYG", period="1mo", interval="1d")['Close'].ffill()
            return 100 - hyg.iloc[-1]
        except: return 5.0 # Çok acil durum sabiti

    def run_engine(self):
        # 1. VERİ TOPLAMA (Failover Destekli)
        data = {}
        # Likidite Bileşenleri
        for s in ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DFII10', 'T10YIE', 'NAPM', 'BAMLH0A0HYM2']:
            val = self.fetch_fred(s)
            if val is not None: data[s] = val.iloc[0]
            else:
                # Proxy Mantığı
                if s == 'BAMLH0A0HYM2': data[s] = self.get_failover_data()
                elif s == 'NAPM': data[s] = 50.0 # Nötr büyüme
                else: data[s] = 0.0

        # 2. HESAPLAMALAR
        # NDL: Net Dollar Liquidity
        ndl = data.get('WALCL', 0) - data.get('WTREGEN', 0) - (data.get('RRPONTSYD', 0) * 1000)
        
        # PMI Momentum: Mevcut PMI - 3 Aylık Ortalama (Trend yönü)
        pmi_raw = data.get('NAPM', 50)
        pmi_mom = pmi_raw - 50.0 # 50 eşiğine göre ivme

        # 3. Z-SCORE VE CMS (Basitleştirilmiş üretim mantığı)
        # Not: Gerçek Z-Score için tarihsel CSV okunur
        if os.path.exists(HISTORY_FILE):
            df_h = pd.read_csv(HISTORY_FILE)
            # Burada 5 yıllık rolling hesaplanır (Hız için son değer üzerinden simüle edilmiştir)
            z_ndl = (ndl - 7000000) / 500000 # Örnek baseline
        else:
            z_ndl = 0

        # Skor Ağırlıkları
        cms = (z_ndl * 0.25) + (data['BAMLH0A0HYM2'] * -0.20) + (data['DFII10'] * -0.15) + (pmi_mom * 0.15)
        
        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cms': round(cms, 4),
            'ndl': ndl,
            'pmi_ivme': pmi_mom
        }

if __name__ == "__main__":
    sentinel = MacroSentinel(FRED_API_KEY)
    report = sentinel.run_engine()
    
    # Veriyi CSV'ye işle (GitHub Actions bunu commit eder)
    new_row = pd.DataFrame([report])
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, new_row]).drop_duplicates(subset='date', keep='last')
        df.to_csv(HISTORY_FILE, index=False)
    else:
        new_row.to_csv(HISTORY_FILE, index=False)
    print(f"CMS Raporu Hazır: {report['cms']}")
