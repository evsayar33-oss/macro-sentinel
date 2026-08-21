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

# Saf NumPy Softmax (Scipy bağımlılığını kaldırmak için)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

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
        except: return None

    def run(self):
        # 1. VERİ TOPLAMA
        fred_series = {
            'fed': 'WALCL', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
            'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 'spread': 'BAMLH0A0HYM2', 
            'tips': 'DFII10'
        }
        
        raw = {}
        for key, s_id in fred_series.items():
            raw[key] = self.fetch_fred(s_id)
            
        y_data = yf.download(["HG=F", "GC=F", "^PCCCE", "ES=F", "DX-Y.NYB"], period="1y", interval="1d", progress=False)['Close'].ffill()
        
        # 2. KOMPOZİT FAKTÖRLER
        # Global Likidite (G3 + NDL)
        try:
            g3_liq = (raw['fed'] + raw['ecb'] + raw['boj']).ffill().iloc[0]
            ndl = raw['fed'].iloc[0] - raw['tga'].iloc[0] - (raw['rrp'].iloc[0] * 1000)
        except:
            g3_liq, ndl = 25000000, 7000000 # Failover
        
        copper_gold = y_data['HG=F'] / y_data['GC=F']
        pc_ratio = y_data['^PCCCE']

        # 3. DİNAMİK IC (Information Coefficient) AĞIRLIKLANDIRMA
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        
        factor_matrix = pd.DataFrame({
            'liq': raw['fed'].reindex(y_data.index, method='ffill'),
            'growth': copper_gold,
            'stress': raw['spread'].reindex(y_data.index, method='ffill'),
            'rates': raw['tips'].reindex(y_data.index, method='ffill')
        }).ffill()
        
        # Son 60 günlük korelasyon gücünü ölç
        corrs = factor_matrix.tail(60).corrwith(spx_ret.tail(60))
        # Softmax ile ağırlıkları dağıt (En çok çalışan faktöre en çok ağırlık)
        weights = softmax(corrs.abs().values)
        
        # 4. SKOR ÜRETİMİ (Z-Score Bazlı)
        def z(val, series): return (val - series.mean()) / (series.std() + 1e-6)
        
        cms = (
            z(ndl, raw['fed']) * weights[0] +
            z(copper_gold.iloc[-1], copper_gold) * weights[1] +
            z(raw['spread'].iloc[-1], raw['spread']) * -weights[2] +
            z(raw['tips'].iloc[-1], raw['tips']) * -weights[3]
        )

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(cms), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(copper_gold.iloc[-1]), 4),
            'pc_ratio': round(float(pc_ratio.iloc[-1]), 2),
            'weights': ",".join([f"{w:.2f}" for w in weights])
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
        print("CMS Başarıyla Güncellendi.")
