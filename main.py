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
        except: 
            return pd.Series() # API Çökerse boş döner, sistem kilitlenmez.

    def run(self):
        # 1. VERİ TOPLAMA (API KALKANI EKLENDİ)
        fred_ids = {
            'fed': 'WALCL', 'ecb': 'ECBASSETSW', 'boj': 'JPNASSETS', 
            'rrp': 'RRPONTSYD', 'tga': 'WTREGEN', 'spread': 'BAMLH0A0HYM2', 
            'tips': 'DFII10', 'pmi': 'NAPM', 'vix': 'VIXCLS',
            'yc': 'T10Y2Y'
        }
        raw = {k: self.fetch_fred(v) for k, v in fred_ids.items()}
        
        # YENİ: TLT (Tahvil ETF'i) eklendi (Korelasyon ölçümü için)
        try:
            y_data = yf.download(["HG=F", "SI=F", "GC=F", "ES=F", "EURUSD=X", "JPYUSD=X", "^VIX", "^VIX3M", "SOXX", "CL=F", "TLT"], period="2y", progress=False)['Close'].ffill()
            api_status = "Online"
        except:
            api_status = "Offline" # API çökerse sistem bunu bilecek
            
        # 2. LİKİDİTE KALİBRASYONU 
        try:
            cur_eur = y_data['EURUSD=X'].iloc[-1]
            cur_jpy = y_data['JPYUSD=X'].iloc[-1]
            fed_m = raw['fed'].iloc[0] 
            ecb_m = raw['ecb'].iloc[0] * cur_eur 
            boj_m = (raw['boj'].iloc[0] * 100) * cur_jpy 
            g3_liq = fed_m + ecb_m + boj_m
            
            tga_val = raw['tga'].iloc[0] if not raw['tga'].empty else 0
            rrp_val = raw['rrp'].iloc[0] if not raw['rrp'].empty else 0
            ndl = fed_m - tga_val - (rrp_val * 1000)
        except: g3_liq, ndl = 21000000, 6800000

        fed_s = raw['fed'].reindex(y_data.index, method='ffill').fillna(method='bfill')
        tga_s = raw['tga'].reindex(y_data.index, method='ffill').fillna(0)
        rrp_s = raw['rrp'].reindex(y_data.index, method='ffill').fillna(0)
        ndl_s = fed_s - tga_s - (rrp_s * 1000)
        spx_ret = y_data['ES=F'].pct_change().shift(-1)
        
        # 3. DARWİNİST TURNUVA VE PETROL
        growth_proxies = {
            'Bakir/Altin': (y_data['HG=F'] / y_data['GC=F']).ffill(),
            'Gumus/Altin': (y_data['SI=F'] / y_data['GC=F']).ffill(),
            'YariIletken/Altin': (y_data['SOXX'] / y_data['GC=F']).ffill()
        }
        best_corr = -1
        active_growth_name = 'Bakir/Altin'
        active_growth_series = growth_proxies['Bakir/Altin']
        for name, series in growth_proxies.items():
            if not series.dropna().empty:
                corr = series.pct_change().tail(90).corr(spx_ret.tail(90))
                if pd.notna(corr) and corr > best_corr:
                    best_corr = corr
                    active_growth_name = name
                    active_growth_series = series

        try:
            oil_series = y_data['CL=F'].tail(30)
            oil_z = (oil_series.iloc[-1] - oil_series.mean()) / (oil_series.std() + 1e-6)
            macro_conf = 1 if active_growth_series.diff(20).iloc[-1] > 0 else -1
            oil_trend = float(oil_z * macro_conf)
        except: oil_trend = 0.0

        # ==========================================
        # YENİ MÜHENDİSLİK: SİYAH KUĞU VE ÇÖKÜŞ SENSÖRÜ
        # ==========================================
        try:
            # 1. Devre Kesici: VIX 35'i geçerse veya 5 günde %40 zıplarsa
            vix_val = raw['vix'].iloc[0] if not raw['vix'].empty else 15.0
            vix_spike = (y_data['^VIX'].iloc[-1] / y_data['^VIX'].iloc[-5]) - 1
            circuit_breaker = (vix_val > 35) or (vix_spike > 0.40)
            
            # 2. Korelasyon 1.0 (Likidite Şoku) Sensörü: S&P, Altın ve Tahvil aynı anda çöküyorsa
            spx_ret_5d = (y_data['ES=F'].iloc[-1] / y_data['ES=F'].iloc[-5]) - 1
            gold_ret_5d = (y_data['GC=F'].iloc[-1] / y_data['GC=F'].iloc[-5]) - 1
            tlt_ret_5d = (y_data['TLT'].iloc[-1] / y_data['TLT'].iloc[-5]) - 1
            
            # Eğer Hisse %3'ten fazla, Altın ve Tahvil %1.5'ten fazla DÜŞÜYORSA -> Margin Call Şoku!
            margin_call_shock = (spx_ret_5d < -0.03) and (gold_ret_5d < -0.015) and (tlt_ret_5d < -0.015)
            
            emergency_mode = circuit_breaker or margin_call_shock
        except:
            emergency_mode = False
        # ==========================================

        # 4. YÜKSELTİLMİŞ DİNAMİK IC MOTORU
        factors = pd.DataFrame({
            'liq_now': ndl_s,                                       
            'liq_fwd': ndl_s.pct_change(60),                        
            'growth_now': active_growth_series, 
            'cycle_fwd': raw['yc'].reindex(y_data.index, method='ffill'), 
            'stress_now': raw['spread'].reindex(y_data.index, method='ffill'), 
            'rates_fwd': raw['tips'].reindex(y_data.index, method='ffill').diff(60) 
        }).ffill().fillna(0).ewm(span=10).mean() 
        
        corrs = factors.tail(90).corrwith(spx_ret.tail(90)).abs().fillna(0)
        weights = softmax(corrs.values) if corrs.sum() > 0.01 else np.array([0.16]*6)

        def z_series(series):
            return (series - series.mean()) / (series.std() + 1e-6) 
        
        pmi_z = raw['pmi'].iloc[0] if not raw['pmi'].empty else 0.0
        pmi_z = (pmi_z - raw['pmi'].mean()) / (raw['pmi'].std() + 1e-6) if not raw['pmi'].empty else 0.0
        
        cms = (z_series(factors['liq_now']).iloc[-1] * weights[0] + 
               z_series(factors['liq_fwd']).iloc[-1] * weights[1] + 
               z_series(factors['growth_now']).iloc[-1] * weights[2] + 
               z_series(factors['cycle_fwd']).iloc[-1] * weights[3] - 
               z_series(factors['stress_now']).iloc[-1] * weights[4] - 
               z_series(factors['rates_fwd']).iloc[-1] * weights[5])
               
        try:
            vix_term = y_data['^VIX'].iloc[-1] / y_data['^VIX3M'].iloc[-1]
        except: vix_term = 0.85 

        # 6. ENSEMBLE (ÇOKLU-VADE) ML AUDITOR
        hist_cms = (z_series(factors['liq_now']) * weights[0] + z_series(factors['liq_fwd']) * weights[1] + 
                    z_series(factors['growth_now']) * weights[2] + z_series(factors['cycle_fwd']) * weights[3] - 
                    z_series(factors['stress_now']) * weights[4] - z_series(factors['rates_fwd']) * weights[5])
        
        strat_returns = hist_cms.shift(1) * spx_ret
        
        sharpe_20 = (strat_returns.tail(20).mean() / (strat_returns.tail(20).std() + 1e-6)) * np.sqrt(252)
        sharpe_60 = (strat_returns.tail(60).mean() / (strat_returns.tail(60).std() + 1e-6)) * np.sqrt(252)
        sharpe_120 = (strat_returns.tail(120).mean() / (strat_returns.tail(120).std() + 1e-6)) * np.sqrt(252)
        
        ensemble_sharpe = (sharpe_20 * 0.20) + (sharpe_60 * 0.50) + (sharpe_120 * 0.30)
        ml_confidence = int(np.clip(50 + (ensemble_sharpe * 25), 10, 100))

        # 7. DİNAMİK PORTFÖY BOYUTLANDIRMA & ACİL DURUM EZİCİSİ (OVERRIDE)
        if emergency_mode:
            # Siyah kuğu gelirse matematik çöpe atılır, sistem %100 Nakite kilitlenir!
            eq_weight = 0
            bond_weight = 0
            cash_weight = 100
            ml_confidence = 0 # Güven skoru sıfırlanır
        else:
            # Normal işleyiş
            eq_raw = 50 + (float(cms) * 25) - ((float(vix_val) - 15) * 1.5)
            eq_weight_primary = np.clip(eq_raw, 0, 100)
            eq_weight = int(eq_weight_primary * (ml_confidence / 100.0))
            bnd_raw = 30 - (float(cms) * 10) + (float(raw['tips'].iloc[0] if not raw['tips'].empty else 1.0) * 5)
            bond_weight = int(np.clip(bnd_raw, 0, 100 - eq_weight))
            cash_weight = 100 - eq_weight - bond_weight

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d %H:%M'),
            'api_status': api_status, # YENİ: Arayüze sistem durumu gönderiliyor
            'emergency': bool(emergency_mode), # YENİ: Çöküş Sensörü
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'active_growth_name': active_growth_name,
            'copper_gold': round(float(y_data['HG=F'].iloc[-1]/y_data['GC=F'].iloc[-1]), 4),
            'vix': round(float(vix_val), 2),
            'real_rate': round(float(raw['tips'].iloc[0]), 2),
            'pmi_z': round(float(pmi_z), 2),
            'yield_curve': round(float(raw['yc'].iloc[0] if not raw['yc'].empty else 0.0), 2),
            'w_str': ",".join([f"{w:.2f}" for w in weights]),
            'vix_term': round(float(vix_term), 3),
            'oil_trend': round(float(oil_trend), 3),
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
            # Tarih formatına saat de eklendiği için subset'i güvenceye alıyoruz
            pd.concat([df_old, df_new]).to_csv(HISTORY_FILE, index=False)
        else:
            df_new.to_csv(HISTORY_FILE, index=False)
        print("Success: Updated with Black Swan & Emergency Override Systems.")
