import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

# --- AYARLAR ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"

class MacroSentinel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_fred(self, s_id, limit=12):
        """FRED'den veri çeker."""
        try:
            params = {'series_id': s_id, 'api_key': self.api_key, 'file_type': 'json', 'sort_order': 'desc', 'limit': limit}
            r = requests.get(self.base_url, params=params, timeout=10).json()
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            return obs['value'].dropna()
        except: return None

    def get_failover_spread(self):
        """FRED Spread hatası verirse Yahoo üzerinden HYG proxy üretir."""
        try:
            hyg = yf.download("HYG", period="5d", progress=False)['Close'].ffill()
            return (100 - hyg.iloc[-1]) / 10 # Normalize edilmiş spread proxy
        except: return 4.0

    def run(self):
        data = {}
        # 1. Veri Toplama
        series = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DFII10', 'T10YIE', 'NAPM', 'BAMLH0A0HYM2']
        for s in series:
            vals = self.fetch_fred(s)
            if vals is not None and not vals.empty:
                if s == 'NAPM': # PMI Momentum: Mevcut - 3 Aylık Ortalama
                    data[s] = vals.iloc[0]
                    data['pmi_avg'] = vals.head(3).mean()
                else:
                    data[s] = vals.iloc[0]
            else:
                # Failover Mantığı
                if s == 'BAMLH0A0HYM2': data[s] = self.get_failover_spread()
                elif s == 'NAPM': data[s], data['pmi_avg'] = 50.0, 50.0
                elif s == 'T10YIE': data[s] = 2.1
                else: data[s] = 0.0

        # 2. Hesaplamalar
        ndl = data.get('WALCL', 0) - data.get('WTREGEN', 0) - (data.get('RRPONTSYD', 0) * 1000)
        pmi_ivme = data.get('NAPM', 50) - data.get('pmi_avg', 50)
        
        # Basitleştirilmiş CMS (Z-Skor simülasyonlu)
        # NDL %25, Spread -%20, Real Rate -%15, PMI Ivme %15, Breakeven %25
        z_ndl = (ndl - 7000000) / 450000
        z_spr = (data.get('BAMLH0A0HYM2', 4) - 4.5) / 1.5
        
        cms = (z_ndl * 0.25) + (z_spr * -0.20) + (data.get('DFII10', 1.5) * -0.15) + (pmi_ivme * 0.15) + (data.get('T10YIE', 2.1) * 0.25)

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cms': round(float(cms), 4),
            'ndl': round(float(ndl), 0),
            'pmi_ivme': round(float(pmi_ivme), 2)
        }

if __name__ == "__main__":
    if not FRED_API_KEY:
        print("HATA: API Key eksik.")
    else:
        engine = MacroSentinel(FRED_API_KEY)
        res = engine.run()
        
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            df_final = pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last')
            df_final.to_csv(HISTORY_FILE, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
        print("İşlem Başarılı.")
