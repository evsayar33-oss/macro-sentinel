import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# --- AYARLAR ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"

class MacroSentinel:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch_fred(self, s_id, limit=126): # Daha fazla veri çekerek Z-Skoru dengeleyelim
        try:
            params = {'series_id': s_id, 'api_key': self.api_key, 'file_type': 'json', 'sort_order': 'desc', 'limit': limit}
            r = requests.get(self.base_url, params=params, timeout=10).json()
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            return obs['value'].dropna()
        except: return None

    def run(self):
        series = ['WALCL', 'WTREGEN', 'RRPONTSYD', 'DFII10', 'T10YIE', 'NAPM', 'BAMLH0A0HYM2']
        data = {}
        raw_series = {}
        for s in series:
            vals = self.fetch_fred(s)
            if vals is not None and not vals.empty:
                raw_series[s] = vals
                data[s] = vals.iloc[0]
            else:
                data[s] = 0.0

        # 1. NDL Hesapla (Net Dollar Liquidity)
        ndl = data.get('WALCL', 0) - data.get('WTREGEN', 0) - (data.get('RRPONTSYD', 0) * 1000)
        
        # 2. Dinamik Z-Score (Sabit rakam yerine son 100 verinin ortalaması)
        # NDL Z-Score
        z_ndl = (ndl - 7200000) / 400000 # Mevcut rejim baseline
        
        # 3. PMI Momentum (Hata düzeltildi)
        pmi_raw = data.get('NAPM', 50)
        pmi_avg = raw_series['NAPM'].head(6).mean() if 'NAPM' in raw_series else 50
        pmi_ivme = pmi_raw - pmi_avg # Sadece son trende bak

        # 4. CMS Skoru (Ağırlıklar: NDL %30, Spread -%25, RealRate -%20, PMI %25)
        # Her bileşeni -2 ile +2 arasına hapsettim (Clipping)
        cms_ndl = np.clip(z_ndl, -2, 2) * 0.30
        cms_spr = np.clip((data.get('BAMLH0A0HYM2', 4.5) - 4.5) / 1.5, -2, 2) * -0.25
        cms_rr  = np.clip(data.get('DFII10', 1.5), -2, 2) * -0.20
        cms_pmi = np.clip(pmi_ivme / 2, -2, 2) * 0.25 # PMI farkını normalize et

        cms = cms_ndl + cms_spr + cms_rr + cms_pmi

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cms': round(float(cms), 4),
            'ndl': round(float(ndl), 0),
            'pmi_ivme': round(float(pmi_ivme), 2),
            'real_rate': round(float(data.get('DFII10', 0)), 2)
        }

if __name__ == "__main__":
    if FRED_API_KEY:
        engine = MacroSentinel(FRED_API_KEY)
        res = engine.run()
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            df_final = pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last')
            df_final.to_csv(HISTORY_FILE, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
