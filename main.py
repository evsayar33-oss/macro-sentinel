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
        fred_ids = {
            'fed': 'WALCL', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
            'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 'spread': 'BAMLH0A0HYM2', 
            'tips': 'DFII10', 'pmi': 'NAPM', 'vix': 'VIXCLS',
            'yc': 'T10Y2Y'
        }
        raw = {k: self.fetch_fred(v) for k, v in fred_ids.items()}
        y_data = yf.download(["HG=F", "GC=F", "ES=F", "EURUSD=X", "JPYUSD=X", "^VIX", "^VIX3M"], period="2y", progress=False)['Close'].ffill()
        
        # 2. G3 LİKİDİTE KALİBRASYONU
        try:
            cur_eur = y_data['EURUSD=X'].iloc[-1]
            cur_jpy = y_data['JPYUSD=X'].iloc[-1]
            fed_m = raw['fed'].iloc[0] 
            ecb_m = raw['ecb'].iloc[0] * cur_eur 
            boj_m = (raw['boj'].iloc[0] * 100) * cur_jpy 
            g3_liq = fed_m + ecb_m + boj_m
            ndl = fed_m - raw['tga'].iloc[0] - (raw['rrp'].iloc[0] * 1000)
        except: g3_liq, ndl = 21000000, 6800000

        fed_s = raw['fed'].reindex(y_data.index, method='ffill')
        tga_s = raw['tga'].reindex(y_data.index, method='ffill').fillna(0)
        rrp_s = raw['rrp'].reindex(y_data.index, method='ffill').fillna(0)
        ndl_s = fed_s - tga_s - (rrp_s * 1000)

        # 3. YÜKSELTİLMİŞ DİNAMİK IC MOTORU
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        
        factors = pd.DataFrame({
            'liq_now': ndl_s,                                       
            'liq_fwd': ndl_s.pct_change(60),                        
            'growth_now': (y_data['HG=F'] / y_data['GC=F']),        
            'cycle_fwd': raw['yc'].reindex(y_data.index, method='ffill'), 
            'stress_now': raw['spread'].reindex(y_data.index, method='ffill'), 
            'rates_fwd': raw['tips'].reindex(y_data.index, method='ffill').diff(60) 
        }).ffill().fillna(0).ewm(span=10).mean() 
        
        corrs = factors.tail(90).corrwith(spx_ret.tail(90)).abs().fillna(0)
        weights = softmax(corrs.values) if corrs.sum() > 0.01 else np.array([0.16]*6)

        # 4. BİRLEŞTİRİLMİŞ (UNIFIED) SKOR ÜRETİMİ
        def z_series(series):
            return (series - series.mean()) / (series.std() + 1e-6)
        
        pmi_z = raw['pmi'].iloc[0] if not raw['pmi'].empty else 0.0
        pmi_z = (pmi_z - raw['pmi'].mean()) / (raw['pmi'].std() + 1e-6) if not raw['pmi'].empty else 0.0
        vix_val = raw['vix'].iloc[0] if not raw['vix'].empty else 15.0
        
        cms = (z_series(factors['liq_now']).iloc[-1] * weights[0] + 
               z_series(factors['liq_fwd']).iloc[-1] * weights[1] + 
               z_series(factors['growth_now']).iloc[-1] * weights[2] + 
               z_series(factors['cycle_fwd']).iloc[-1] * weights[3] - 
               z_series(factors['stress_now']).iloc[-1] * weights[4] - 
               z_series(factors['rates_fwd']).iloc[-1] * weights[5])
               
        try:
            vix_term = y_data['^VIX'].iloc[-1] / y_data['^VIX3M'].iloc[-1]
        except: vix_term = 0.85 

        # 5. YENİ: ML AUDITOR (GÖLGE DENETÇİ)
        # Geçmiş 60 günün CMS skorlarını simüle edip Sharpe Rasyosunu buluyoruz.
        hist_cms = (z_series(factors['liq_now']) * weights[0] + z_series(factors['liq_fwd']) * weights[1] + 
                    z_series(factors['growth_now']) * weights[2] + z_series(factors['cycle_fwd']) * weights[3] - 
                    z_series(factors['stress_now']) * weights[4] - z_series(factors['rates_fwd']) * weights[5])
        
        strat_returns = hist_cms.shift(1) * spx_ret
        recent_sharpe = (strat_returns.tail(60).mean() / (strat_returns.tail(60).std() + 1e-6)) * np.sqrt(252)
        
        # Sharpe rasyosunu 0-100 arası bir "Güven Skoruna" çeviriyoruz
        ml_confidence = int(np.clip(50 + (recent_sharpe * 25), 10, 100))

        # 6. DİNAMİK PORTFÖY BOYUTLANDIRMA & ML MÜDAHALESİ
        # Ana modelin önerdiği hisse ağırlığı
        eq_raw = 50 + (float(cms) * 25) - ((float(vix_val) - 15) * 1.5)
        eq_weight_primary = np.clip(eq_raw, 0, 100)
        
        # ML Denetçisi devreye giriyor (Güven düşükse hisse ağırlığını tıraşlar)
        eq_weight = int(eq_weight_primary * (ml_confidence / 100.0))
        
        bnd_raw = 30 - (float(cms) * 10) + (float(raw['tips'].iloc[0] if not raw['tips'].empty else 1.0) * 5)
        bond_weight = int(np.clip(bnd_raw, 0, 100 - eq_weight))
        cash_weight = 100 - eq_weight - bond_weight

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d'),
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'copper_gold': round(float(y_data['HG=F'].iloc[-1]/y_data['GC=F'].iloc[-1]), 4),
            'vix': round(float(vix_val), 2),
            'real_rate': round(float(raw['tips'].iloc[0]), 2),
            'pmi_z': round(float(pmi_z), 2),
            'yield_curve': round(float(raw['yc'].iloc[0] if not raw['yc'].empty else 0.0), 2),
            'w_str': ",".join([f"{w:.2f}" for w in weights]),
            'vix_term': round(float(vix_term), 3),
            'ml_confidence': ml_confidence,
            'eq_weight': eq_weight,
            'bond_weight': bond_weight,
            'cash_weight': cash_weight
        }

if __name__ == "__main__":
    if FRED_API_KEY:
        engine = UltimateSentinelEngine(FRED_API_KEY)
        res = engine.run()
        df_new = pd.DataFrame([res])
        if os.path.exists(HISTORY_FILE):
            df_old = pd.read_csv(HISTORY_FILE)
            pd.concat([df_old, df_new]).drop_duplicates(subset='date', keep='last').to_csv(HISTORY_FILE, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
        print("Success: Ultimate Pro Quant CMS Updated with ML Auditor.")
