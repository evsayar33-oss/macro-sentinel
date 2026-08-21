import os
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

# --- AYARLAR ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"

class UltimateSentinelEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_fred(self, s_id, limit=252):
        try:
            params = {'series_id': s_id, 'api_key': self.api_key, 'file_type': 'json', 'sort_order': 'desc', 'limit': limit}
            r = requests.get(self.base_url, params=params, timeout=15).json()
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            return obs.set_index(pd.to_datetime(obs['date']))['value'].dropna()
        except: return pd.Series()

    def run(self):
        # 1. VERİ TOPLAMA
        fred_series = {
            'fed': 'WALCL', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
            'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 'spread': 'BAMLH0A0HYM2', 
            'tips': 'DFII10', 'pmi': 'NAPM'
        }
        raw = {k: self.fetch_fred(s_id) for k, s_id in fred_series.items()}
            
        # Kurlar ve High-Frequency Veriler
        y_data = yf.download(["HG=F", "GC=F", "^PCCCE", "ES=F", "EURUSD=X", "JPYUSD=X"], period="1y", progress=False)['Close'].ffill()
        
        # 2. KUR DÜZELTMELİ GLOBAL LİKİDİTE (G3)
        try:
            fed_usd = raw['fed'].iloc[0] # Milyon $
            ecb_usd = raw['ecb'].iloc[0] * y_data['EURUSD=X'].iloc[-1] # Milyon Euro -> USD
            boj_usd = (raw['boj'].iloc[0] * 1000) * y_data['JPYUSD=X'].iloc[-1] # Milyar Yen -> USD (Milyona çevrildi)
            g3_liq = fed_usd + ecb_usd + boj_usd
            
            ndl = raw['fed'].iloc[0] - raw['tga'].iloc[0] - (raw['rrp'].iloc[0] * 1000)
        except: g3_liq, ndl = 25000000, 7000000 # Failover

        # 3. PMI MOMENTUM (nan korumalı)
        pmi_vals = raw['pmi']
        pmi_ivme = pmi_vals.iloc[0] - pmi_vals.head(6).mean() if len(pmi_vals) > 1 else 0.0

        # 4. SKOR ÜRETİMİ
        # NDL Z-Score (6.8M baseline)
        z_ndl = (ndl - 6800000) / 450000
        
        # Dinamik CMS (Ağırlıklar: NDL %30, Spread %25, Reel Rate %20, PMI %25)
        cms = (np.clip(z_ndl, -2, 2) * 0.30 + 
               np.clip((raw['spread'].iloc[0] - 4.5)/1.5, -2, 2) * -0.25 + 
               np.clip(raw['tips'].iloc[0], -2, 2) * -0.20 + 
               np.clip(pmi_ivme / 2, -2, 2) * 0.25)

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(cms), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(y_data['HG=F'].iloc[-1] / y_data['GC=F'].iloc[-1]), 4),
            'pc_ratio': round(float(y_data['^PCCCE'].iloc[-1]), 2),
            'real_rate': round(float(raw['tips'].iloc[0]), 2),
            'pmi_ivme': round(float(pmi_ivme), 2)
        }

if __name__ == "__main__":
    if FRED_API_KEY:
        engine = UltimateSentinelEngine(FRED_API_KEY)
        res = engine.run()
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            df_final = pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last')
            df_final.to_csv(HISTORY_FILE, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
