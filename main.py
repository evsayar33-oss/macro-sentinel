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

    def fetch_fred(self, s_id, limit=500):
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
            'tips': 'DFII10'
        }
        
        raw = {}
        for key, s_id in fred_series.items():
            raw[key] = self.fetch_fred(s_id)
            
        y_data = yf.download(["HG=F", "GC=F", "^PCCCE", "ES=F"], period="2y", interval="1d", progress=False)['Close'].ffill()
        
        # 2. HİZALAMA VE G3 HESAPLAMA (Kritik NaN Koruması)
        fed = raw['fed']
        ecb = raw['ecb'].reindex(fed.index, method='ffill')
        boj = raw['boj'].reindex(fed.index, method='ffill')
        
        g3_liq_series = fed.add(ecb, fill_value=0).add(boj, fill_value=0)
        g3_liq = g3_liq_series.iloc[0]
        
        ndl = raw['fed'].iloc[0] - raw['tga'].reindex(fed.index, method='ffill').iloc[0] - (raw['rrp'].reindex(fed.index, method='ffill').iloc[0] * 1000)
        
        copper_gold = (y_data['HG=F'] / y_data['GC=F']).ffill()
        pc_ratio = y_data['^PCCCE'].ffill().fillna(0.7)

        # 3. DİNAMİK IC AĞIRLIKLANDIRMA
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        factor_matrix = pd.DataFrame({
            'liq': raw['fed'].reindex(y_data.index, method='ffill'),
            'growth': copper_gold,
            'stress': raw['spread'].reindex(y_data.index, method='ffill'),
            'rates': raw['tips'].reindex(y_data.index, method='ffill')
        }).ffill().fillna(0)
        
        corrs = factor_matrix.tail(90).corrwith(spx_ret.tail(90)).fillna(0)
        weights = softmax(corrs.abs().values) if corrs.abs().sum() > 0 else np.array([0.25, 0.25, 0.25, 0.25])
        
        # 4. SKOR ÜRETİMİ
        def z(val, series): 
            if series.std() == 0: return 0
            return (val - series.mean()) / (series.std() + 1e-6)
        
        cms = (
            z(ndl, raw['fed']) * weights[0] +
            z(copper_gold.iloc[-1], copper_gold) * weights[1] +
            z(raw['spread'].iloc[0], raw['spread']) * -weights[2] +
            z(raw['tips'].iloc[0], raw['tips']) * -weights[3]
        )

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(copper_gold.iloc[-1]), 4),
            'pc_ratio': round(float(pc_ratio.iloc[-1]), 2),
            'weights': ",".join([f"{w:.2f}" for w in weights]),
            'real_rate': round(float(raw['tips'].iloc[0]), 2)
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
