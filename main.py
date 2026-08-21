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

class SovereignMacroEngine:
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
        y_data = yf.download(["HG=F", "GC=F", "ES=F", "EURUSD=X", "JPYUSD=X", "^VIX"], period="2y", progress=False)['Close'].ffill()
        
        # 2. G3 LİKİDİTE KALİBRASYONU (KURUMSAL BİRİM DÜZELTME)
        # Fed: Millions, ECB: Millions EUR, BoJ: 100 Millions Yen (FRED standard)
        try:
            fed_m = raw['fed'].iloc[0]
            ecb_m = raw['ecb'].iloc[0] * y_data['EURUSD=X'].iloc[-1]
            # BoJ: 100M Yen birimini Milyon USD'ye çevir: (Değer * 100) / (Yen/USD Kuru)
            # y_data['JPYUSD=X'] 1 Yen'in kaç Dolar olduğunu verir (örn 0.0069)
            boj_m = (raw['boj'].iloc[0] * 100) * y_data['JPYUSD=X'].iloc[-1]
            g3_liq = fed_m + ecb_m + boj_m
            
            ndl = raw['fed'].iloc[0] - raw['tga'].iloc[0] - (raw['rrp'].iloc[0] * 1000)
        except: g3_liq, ndl = 21000000, 6800000 # Gerçekçi Global Failover

        # 3. DİNAMİK IC (KORELASYON) MOTORU
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        factors = pd.DataFrame({
            'liq': raw['fed'].reindex(y_data.index, method='ffill'),
            'growth': (y_data['HG=F'] / y_data['GC=F']),
            'stress': raw['spread'].reindex(y_data.index, method='ffill'),
            'rates': raw['tips'].reindex(y_data.index, method='ffill')
        }).ffill()
        
        # Sinyal boşluklarını temizle ve korelasyonu hesapla
        corrs = factors.tail(90).corrwith(spx_ret.tail(90)).fillna(0).abs()
        # Eğer veri çok yeniyse eşit dağıt, yoksa dinamik ağırlıklandır
        weights = softmax(corrs.values) if corrs.sum() > 0.01 else np.array([0.25, 0.25, 0.25, 0.25])

        # 4. SKOR ÜRETİMİ (Z-SCORE)
        def z(val, series): return (val - series.mean()) / (series.std() + 1e-6)
        
        pmi_z = z(raw['pmi'].iloc[0], raw['pmi']) if not raw['pmi'].empty else 0
        copper_gold = y_data['HG=F'] / y_data['GC=F']
        
        cms = (z(ndl, raw['fed']) * weights[0] + 
               z(copper_gold.iloc[-1], copper_gold) * weights[1] + 
               z(raw['spread'].iloc[0], raw['spread']) * -weights[2] + 
               z(raw['tips'].iloc[0], raw['tips']) * -weights[3])

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(copper_gold.iloc[-1]), 4),
            'vix': round(float(y_data['^VIX'].iloc[-1]), 2),
            'real_rate': round(float(raw['tips'].iloc[0]), 2),
            'pmi_z': round(float(pmi_z), 2),
            'weights_json': ",".join([f"{w:.2f}" for w in weights])
        }

if __name__ == "__main__":
    if FRED_API_KEY:
        res = SovereignMacroEngine(FRED_API_KEY).run()
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last').to_csv(HISTORY_FILE, index=False)
        else: df_new.to_csv(HISTORY_FILE, index=False)
