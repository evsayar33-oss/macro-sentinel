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

    def fetch_fred(self, s_id, limit=12):
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
        for s in series:
            vals = self.fetch_fred(s)
            if vals is not None and not vals.empty:
                data[s] = vals.iloc[0]
                if s == 'NAPM': data['pmi_avg'] = vals.head(3).mean()
            else:
                data[s] = 0.0

        # Hesaplamalar
        ndl = data.get('WALCL', 0) - data.get('WTREGEN', 0) - (data.get('RRPONTSYD', 0) * 1000)
        pmi_ivme = data.get('NAPM', 50) - data.get('pmi_avg', 50)
        
        # CMS Skoru (Z-Score simülasyonu)
        z_ndl = (ndl - 6800000) / 400000
        z_spr = (data.get('BAMLH0A0HYM2', 4.5) - 4.5) / 1.5
        cms = (z_ndl * 0.30) + (z_spr * -0.25) + (data.get('DFII10', 1.5) * -0.20) + (pmi_ivme * 0.25)

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cms': round(float(cms), 4),
            'ndl': round(float(ndl), 0),
            'pmi_ivme': round(float(pmi_ivme), 2),
            'real_rate': round(float(data.get('DFII10', 0)), 2), # YENİ SÜTUN
            'spread': round(float(data.get('BAMLH0A0HYM2', 0)), 2) # YENİ SÜTUN
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
