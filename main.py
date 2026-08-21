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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class UltimateSentinelEngine:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_fred(self, s_id, limit=300):
        try:
            params = {'series_id': s_id, 'api_key': self.api_key, 'file_type': 'json', 'sort_order': 'desc', 'limit': limit}
            r = requests.get(self.base_url, params=params, timeout=15).json()
            if 'observations' not in r: return pd.Series()
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            return obs.set_index(pd.to_datetime(obs['date']))['value'].dropna()
        except: return pd.Series()

    def run(self):
        # 1. VERİ TOPLAMA
        fred_ids = {'fed': 'WALCL', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
                    'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 'spread': 'BAMLH0A0HYM2', 
                    'tips': 'DFII10', 'pmi': 'NAPM'}
        raw = {k: self.fetch_fred(v) for k, v in fred_ids.items()}
            
        y_data = yf.download(["HG=F", "GC=F", "ES=F", "EURUSD=X", "JPYUSD=X", "^VIX", "GSG"], 
                             period="2y", progress=False)['Close'].ffill()
        
        # 2. GLOBAL LİKİDİTE (Birim Kalibrasyonu)
        fed_m = raw['fed'].iloc[0] if not raw['fed'].empty else 7200000
        ecb_m = (raw['ecb'].iloc[0] * y_data['EURUSD=X'].iloc[-1]) if not raw['ecb'].empty else 6000000
        # BoJ Bilançosu Milyar Yen gelir. Milyon USD için: (Değer * 1000) * Kur
        boj_m = (raw['boj'].iloc[0] * 1000 * y_data['JPYUSD=X'].iloc[-1]) if not raw['boj'].empty else 9000000
        g3_liq = fed_m + ecb_m + boj_m
        
        tga = raw['tga'].iloc[0] if not raw['tga'].empty else 700000
        rrp = raw['rrp'].iloc[0] if not raw['rrp'].empty else 400000
        ndl = fed_m - tga - (rrp * 1000)

        # 3. DİNAMİK IC VE PMI MOMENTUM
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        factors = pd.DataFrame({
            'liq': raw['fed'].reindex(y_data.index, method='ffill') if not raw['fed'].empty else 0,
            'growth': (y_data['HG=F'] / y_data['GC=F']),
            'stress': raw['spread'].reindex(y_data.index, method='ffill') if not raw['spread'].empty else 0,
            'rates': raw['tips'].reindex(y_data.index, method='ffill') if not raw['tips'].empty else 0
        }).ffill().fillna(0)
        
        corrs = factors.tail(60).corrwith(spx_ret.tail(60)).fillna(0).abs()
        weights = softmax(corrs.values) if corrs.sum() > 0 else np.array([0.25, 0.25, 0.25, 0.25])

        def z(val, series):
            if series.empty or series.std() == 0: return 0.0
            return (val - series.mean()) / (series.std() + 1e-6)
        
        # PMI Büyüme İvmesi (Cari Değer - 12 Aylık Ortalama)
        pmi_ivme = raw['pmi'].iloc[0] - raw['pmi'].head(12).mean() if not raw['pmi'].empty else 0.0
        
        # 4. SKOR ÜRETİMİ
        cms = (z(ndl, raw['fed']) * weights[0] + 
               z(y_data['HG=F'].iloc[-1] / y_data['GC=F'].iloc[-1], (y_data['HG=F'] / y_data['GC=F'])) * weights[1] + 
               z(raw['spread'].iloc[0], raw['spread']) * -weights[2] + 
               z(raw['tips'].iloc[0], raw['tips']) * -weights[3])

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(y_data['HG=F'].iloc[-1] / y_data['GC=F'].iloc[-1]), 4),
            'vix': round(float(y_data['^VIX'].iloc[-1]), 2),
            'real_rate': round(float(raw['tips'].iloc[0]), 2),
            'pmi_growth': round(float(pmi_ivme), 2),
            'w_str': ",".join([f"{w:.2f}" for w in weights])
        }

if __name__ == "__main__":
    if FRED_API_KEY:
        res = UltimateSentinelEngine(FRED_API_KEY).run()
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last').to_csv(HISTORY_FILE, index=False)
        else: df_new.to_csv(HISTORY_FILE, index=False)
