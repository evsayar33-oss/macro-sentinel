import os
import requests
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import pytz

from regime_engine import MacroRegimeEngine

# --- AYARLAR ---
FRED_API_KEY = os.getenv("FRED_API_KEY")
HISTORY_FILE = "cms_history.csv"


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum() + 1e-6)


class UltimateSentinelEngine:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        self.regime_engine = MacroRegimeEngine()

    def fetch_fred(self, s_id, limit=350):
        if not self.api_key:
            return pd.Series(dtype=float)
        try:
            params = {
                'series_id': s_id,
                'api_key': self.api_key,
                'file_type': 'json',
                'sort_order': 'desc',
                'limit': limit
            }
            r = requests.get(self.base_url, params=params, timeout=15).json()
            if 'observations' not in r:
                return pd.Series(dtype=float)
            obs = pd.DataFrame(r['observations'])[['date', 'value']]
            obs['value'] = pd.to_numeric(obs['value'], errors='coerce')
            s = obs.set_index(pd.to_datetime(obs['date']))['value'].dropna()
            return s.sort_index()
        except:
            return pd.Series(dtype=float)

    def run(self):
        # 1. VERİ TOPLAMA (Genişletilmiş Makro ve Piyasa Göstergeleri)
        fred_ids = {
            'fed': 'WALCL',
            'ecb': 'ECBASSETSW',
            'boj': 'JPNASSETS',
            'rrp': 'RRPONTSYD',
            'tga': 'WTREGEN',
            'spread': 'BAMLH0A0HYM2',
            'tips': 'DFII10',
            'pmi': 'NAPM',
            'vix': 'VIXCLS',
            'yc': 'T10Y2Y',
            't10yie': 'T10YIE',
            'dxy': 'DTWEXBGS',
            'ig_spread': 'BAMLC0A0CM',
            'dgs2': 'DGS2',
            'dgs10': 'DGS10'
        }
        raw = {k: self.fetch_fred(v) for k, v in fred_ids.items()}

        tickers = [
            "HG=F", "SI=F", "GC=F", "ES=F", "EURUSD=X", "USDJPY=X",
            "JPYUSD=X", "^VIX", "^VIX3M", "SOXX", "CL=F", "TLT", "BTC-USD", "BDRY"
        ]
        try:
            y_data = yf.download(tickers, period="2y", progress=False)['Close'].ffill()
            if y_data.empty:
                api_status = "Offline"
            else:
                api_status = "Online"
        except:
            api_status = "Offline"
            y_data = pd.DataFrame()

        # Fallback sentetik / varsayılan endeks eğer API'ler erişilemezse
        if y_data.empty or 'ES=F' not in y_data.columns:
            # Fallback mock frame son 300 iş günü için
            b_dates = pd.date_range(end=datetime.now(), periods=300, freq='B')
            y_data = pd.DataFrame(index=b_dates)
            y_data['ES=F'] = 5400.0
            y_data['GC=F'] = 2500.0
            y_data['HG=F'] = 4.2
            y_data['SI=F'] = 29.0
            y_data['SOXX'] = 220.0
            y_data['CL=F'] = 75.0
            y_data['TLT'] = 95.0
            y_data['EURUSD=X'] = 1.09
            y_data['USDJPY=X'] = 145.0
            y_data['JPYUSD=X'] = 0.0069
            y_data['^VIX'] = 15.5
            y_data['^VIX3M'] = 18.0
            y_data['BTC-USD'] = 58000.0
            y_data['BDRY'] = 8.5

        # 2. LİKİDİTE KALİBRASYONU (G3 ve Net Dolar Likiditesi)
        try:
            cur_eur = y_data['EURUSD=X'].iloc[-1] if 'EURUSD=X' in y_data else 1.08
            cur_jpy = (1.0 / y_data['USDJPY=X'].iloc[-1]) if ('USDJPY=X' in y_data and y_data['USDJPY=X'].iloc[-1] > 0) else y_data.get('JPYUSD=X', pd.Series([0.0067])).iloc[-1]
            fed_m = raw['fed'].iloc[-1] if not raw['fed'].empty else 7200000.0
            ecb_m = (raw['ecb'].iloc[-1] if not raw['ecb'].empty else 6500000.0) * cur_eur
            boj_m = ((raw['boj'].iloc[-1] if not raw['boj'].empty else 7500000.0) * 100) * cur_jpy
            g3_liq = fed_m + ecb_m + boj_m

            tga_val = raw['tga'].iloc[-1] if not raw['tga'].empty else 750000.0
            rrp_val = raw['rrp'].iloc[-1] if not raw['rrp'].empty else 300.0
            ndl = fed_m - tga_val - (rrp_val * 1000)
        except:
            g3_liq, ndl = 17800000.0, 5768000.0

        # Zaman serileri hizalaması
        def safe_reindex(s, default_val=0.0):
            if s.empty:
                return pd.Series(default_val, index=y_data.index)
            return s.reindex(y_data.index, method='ffill').fillna(method='bfill').fillna(default_val)

        fed_s = safe_reindex(raw['fed'], 7200000.0)
        tga_s = safe_reindex(raw['tga'], 750000.0)
        rrp_s = safe_reindex(raw['rrp'], 300.0)
        ndl_s = fed_s - tga_s - (rrp_s * 1000)
        spx_ret = y_data['ES=F'].pct_change().shift(-1)

        # 3. BÜYÜME GÖSTERGELERİ VE PETROL TRENDİ
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
        except:
            oil_trend = 0.0

        # 4. KLASİK CMS FAKTÖR SKORU
        tips_s = safe_reindex(raw['tips'], 2.0)
        yc_s = safe_reindex(raw['yc'], 0.2)
        spread_s = safe_reindex(raw['spread'], 3.5)

        factors = pd.DataFrame({
            'liq_now': ndl_s,
            'liq_fwd': ndl_s.pct_change(60),
            'growth_now': active_growth_series,
            'cycle_fwd': yc_s,
            'stress_now': spread_s,
            'rates_fwd': tips_s.diff(60)
        }).ffill().fillna(0).ewm(span=10).mean()

        corrs = factors.tail(90).corrwith(spx_ret.tail(90)).abs().fillna(0)
        weights = softmax(corrs.values) if corrs.sum() > 0.01 else np.array([0.16] * 6)

        def z_series(series):
            return (series - series.mean()) / (series.std() + 1e-6)

        cms = (z_series(factors['liq_now']).iloc[-1] * weights[0] +
               z_series(factors['liq_fwd']).iloc[-1] * weights[1] +
               z_series(factors['growth_now']).iloc[-1] * weights[2] +
               z_series(factors['cycle_fwd']).iloc[-1] * weights[3] -
               z_series(factors['stress_now']).iloc[-1] * weights[4] -
               z_series(factors['rates_fwd']).iloc[-1] * weights[5])

        try:
            vix_term = y_data['^VIX'].iloc[-1] / y_data['^VIX3M'].iloc[-1]
        except:
            vix_term = 0.85

        # =========================================================================
        # 5. YENİ MODÜL: MACRO EVENT INTERPRETATION SYSTEM (5 DETERMINISTIC REGIMES)
        # =========================================================================
        # Construct DataFrame containing all indicator time series
        macro_df = pd.DataFrame(index=y_data.index)
        macro_df['oil'] = y_data.get('CL=F', pd.Series(75.0, index=y_data.index))
        macro_df['freight'] = y_data.get('BDRY', pd.Series(8.5, index=y_data.index))
        macro_df['hy_oas'] = safe_reindex(raw['spread'], 3.5)
        macro_df['ig_oas'] = safe_reindex(raw['ig_spread'], 1.1)
        macro_df['spx'] = y_data.get('ES=F', pd.Series(5400.0, index=y_data.index))
        macro_df['ust10y'] = safe_reindex(raw['dgs10'], 4.0)
        macro_df['broad_dollar'] = safe_reindex(raw['dxy'], 120.0)
        macro_df['usdjpy'] = y_data.get('USDJPY=X', pd.Series(145.0, index=y_data.index))
        macro_df['vix'] = safe_reindex(raw['vix'], y_data.get('^VIX', pd.Series(15.0, index=y_data.index)).iloc[-1])
        macro_df['btc'] = y_data.get('BTC-USD', pd.Series(58000.0, index=y_data.index))
        macro_df['tips10y'] = safe_reindex(raw['tips'], 2.0)
        macro_df['t10yie'] = safe_reindex(raw['t10yie'], 2.2)
        macro_df['dgs2'] = safe_reindex(raw['dgs2'], 4.0)
        macro_df['dgs10'] = safe_reindex(raw['dgs10'], 4.2)
        macro_df['ndl'] = ndl_s
        macro_df['gold'] = y_data.get('GC=F', pd.Series(2500.0, index=y_data.index))

        prepared_macro = self.regime_engine.prepare_indicators(macro_df)

        # Retrieve calibrated optimal thresholds from config if available
        calibrated_th = self.regime_engine.config.get("calibrated_thresholds", None)

        # Run time series evaluation to obtain deterministic state & 2-week hysteresis
        classified_macro = self.regime_engine.run_time_series(prepared_macro, custom_thresholds=calibrated_th)
        latest_macro = classified_macro.iloc[-1]

        regime_id = int(latest_macro['confirmed_regime_id'])
        regime_name = str(latest_macro['confirmed_regime_name'])
        regime_type = str(latest_macro['regime_type'])
        regime_subtype = str(latest_macro['regime_subtype'])
        h_days_left = int(latest_macro['hysteresis_days_left'])
        conflict_note = str(latest_macro['conflict_note'])

        # Hysteresis display status
        if regime_id == 0:
            regime_status = "REJİMSİZ GEÇİŞ (Nötr / Dengeli Portföy)"
        elif h_days_left > 0:
            regime_status = f"Teyit Edildi (Histerezis Koruması Aktif: {h_days_left} gün)"
        else:
            regime_status = "Teyit Edildi (Aktif Şok/Ralli)"

        # Portfolio sizing from deterministic macro regime
        reg_weights = self.regime_engine.get_portfolio_weights(regime_id, regime_subtype)
        eq_weight = reg_weights['equity']
        bond_weight = reg_weights['bond']
        cash_weight = reg_weights['cash']

        # Siyah Kuğu / Emergency Override (Regime 2 veya Volatilite Şoku)
        emergency_mode = bool(regime_id == 2 or (float(latest_macro.get('vix', 15.0)) > 35.0))
        if emergency_mode:
            eq_weight = 0
            bond_weight = 10
            cash_weight = 90
            ml_confidence = 0
        else:
            ml_confidence = int(np.clip(75 + float(cms) * 20, 20, 100))

        vix_val = float(raw['vix'].iloc[-1]) if not raw['vix'].empty else float(latest_macro.get('vix', 15.0))
        tips_val = float(raw['tips'].iloc[-1]) if not raw['tips'].empty else float(latest_macro.get('tips10y', 2.0))
        pmi_val = float(raw['pmi'].iloc[-1]) if not raw['pmi'].empty else 50.0
        pmi_z = (pmi_val - 50.0) / 3.0
        yc_val = float(raw['yc'].iloc[-1]) if not raw['yc'].empty else float(latest_macro.get('dgs10', 4.0) - latest_macro.get('dgs2', 4.0))

        return {
            'date': datetime.now(pytz.timezone('Europe/Istanbul')).strftime('%Y-%m-%d %H:%M'),
            'api_status': api_status,
            'emergency': bool(emergency_mode),
            'cms': round(float(np.nan_to_num(cms)), 4),
            'ndl': round(float(ndl), 0),
            'g3_liq': round(float(g3_liq), 0),
            'active_growth_name': active_growth_name,
            'copper_gold': round(float(y_data['HG=F'].iloc[-1] / y_data['GC=F'].iloc[-1]), 4),
            'vix': round(float(vix_val), 2),
            'real_rate': round(float(tips_val), 2),
            'pmi_z': round(float(pmi_z), 2),
            'yield_curve': round(float(yc_val), 2),
            'w_str': ",".join([f"{w:.2f}" for w in weights]),
            'vix_term': round(float(vix_term), 3),
            'oil_trend': round(float(oil_trend), 3),
            'ml_confidence': ml_confidence,
            'eq_weight': eq_weight,
            'bond_weight': bond_weight,
            'cash_weight': cash_weight,
            # YENİ ENTEGRASYON: 5 DETERMINISTIC MAKRO REJİM PARAMETRELERİ
            'regime_id': regime_id,
            'regime_name': regime_name,
            'regime_type': regime_type,
            'regime_subtype': regime_subtype,
            'regime_status': regime_status,
            'hysteresis_days_left': h_days_left,
            'conflict_note': conflict_note,
            'oil_ret_20d_z': round(float(latest_macro.get('oil_ret_20d_z', 0.0)), 2),
            'freight_lvl_z': round(float(latest_macro.get('freight_lvl_z', 0.0)), 2),
            'hy_oas_z': round(float(latest_macro.get('hy_oas_z', 0.0)), 2),
            'spx_bond_corr': round(float(latest_macro.get('spx_bond_corr_60d', 0.0)), 2),
            'broad_dollar_5d_z': round(float(latest_macro.get('broad_dollar_5d_z', 0.0)), 2),
            'broad_dollar_z': round(float(latest_macro.get('broad_dollar_z', 0.0)), 2),
            'usdjpy_1d_z': round(float(latest_macro.get('usdjpy_1d_z', 0.0)), 2),
            'vix_z': round(float(latest_macro.get('vix_z', 0.0)), 2),
            'vix_percentile_252': round(float(latest_macro.get('vix_percentile_252', 50.0)), 1),
            'tips_1d_z': round(float(latest_macro.get('tips_1d_z', 0.0)), 2),
            't10yie_z': round(float(latest_macro.get('t10yie_z', 0.0)), 2),
            'ndl_z': round(float(latest_macro.get('ndl_z', 0.0)), 2)
        }


if __name__ == "__main__":
    engine = UltimateSentinelEngine(FRED_API_KEY)
    res = engine.run()
    df_new = pd.DataFrame([res])

    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        # Ensure all new columns exist in old dataframe
        for col in df_new.columns:
            if col not in df_old.columns:
                df_old[col] = np.nan
        pd.concat([df_old, df_new], ignore_index=True).to_csv(HISTORY_FILE, index=False)
    else:
        df_new.to_csv(HISTORY_FILE, index=False)

    print(f"Success: Macro Event Interpretation System active. Confirmed Regime {res['regime_id']}: {res['regime_name']} ({res['regime_subtype']}).")
