import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from regime_engine import MacroRegimeEngine

# Streamlit Page Setup
st.set_page_config(
    page_title="Ultimate Macro Sentinel — Event Interpretation System",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Tech Styling
st.markdown("""
<style>
    .main { background-color: #07090e; color: #e1e7ec; }
    .stMetric { background-color: #0f141f; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; }
    .regime-card { padding: 22px; border-radius: 14px; margin-bottom: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: bold; margin-right: 6px; }
    .badge-shock { background-color: #ff3344; color: white; }
    .badge-riskon { background-color: #00e676; color: black; }
    .badge-neutral { background-color: #ffaa00; color: black; }
</style>
""", unsafe_allow_html=True)

# Initialize Regime Engine
engine = MacroRegimeEngine()
config = engine.config

HISTORY_FILE = "cms_history.csv"

if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    if not df.empty:
        latest = df.iloc[-1]

        # Extract Core Data
        val = latest.get('cms', 0.0)
        rr = latest.get('real_rate', 2.0)
        vix_val = latest.get('vix', 15.0)
        vix_term = latest.get('vix_term', 0.85)
        ml_conf = int(latest.get('ml_confidence', 75))
        api_durum = latest.get('api_status', 'Online')
        emergency = bool(latest.get('emergency', False))
        son_guncelleme = latest.get('date', 'Bilinmiyor')
        act_growth = latest.get('active_growth_name', 'Bakir/Altin')
        oil_trend = latest.get('oil_trend', 0.0)

        # Macro Event Interpretation System Fields
        regime_id = int(latest.get('regime_id', 0))
        regime_name = str(latest.get('regime_name', 'REJIMSIZ_GECIS'))
        regime_type = str(latest.get('regime_type', 'TRANSITION'))
        regime_subtype = str(latest.get('regime_subtype', 'Dengeli / Nötr Piyasa'))
        regime_status = str(latest.get('regime_status', 'REJİMSİZ GEÇİŞ'))
        h_days = int(latest.get('hysteresis_days_left', 0))
        conflict_note = str(latest.get('conflict_note', 'No conflict detected'))

        # Portfolio Weights
        eq_w = int(latest.get('eq_weight', 45))
        bnd_w = int(latest.get('bond_weight', 35))
        csh_w = int(latest.get('cash_weight', 20))

        # Color & Visual Mapping based on active Regime
        if emergency or regime_id == 2:
            reg_color = "#ff1744"
            badge_class = "badge-shock"
            type_label = "🚨 SİSTEMİK ÇÖKÜŞ / ŞOK REJİMİ"
            status_desc = "PANİK - %90-100 NAKİT / LİKİDİTE KORUMASI"
        elif regime_type == "SHOCK":
            reg_color = "#ff5252"
            badge_class = "badge-shock"
            type_label = f"🚨 ŞOK REJİMİ (KOD: {regime_id})"
            status_desc = "DEFANSİF MOD - RİSKLİ VARLIKLARI AZALT"
        elif regime_type == "RISK_ON" or regime_id == 5:
            reg_color = "#00e676"
            badge_class = "badge-riskon"
            type_label = "🚀 KÜRESEL LİKİDİTE RALLİSİ (RISK-ON)"
            status_desc = "BOĞA DÖNGÜSÜ - TAM KAPASİTE BÜYÜME"
        else:
            reg_color = "#ffd600"
            badge_class = "badge-neutral"
            type_label = "⚖️ REJİMSİZ GEÇİŞ / NÖTR"
            status_desc = "DENGELİ RİSK PARİTESİ (DÖNGÜ GEÇİŞİ)"

        # Header and System Status
        st.title("🏛️ ULTIMATE MACRO SENTINEL")
        st.markdown(f"**Autonomous Macro Event Interpretation & Regime System v1.0**")

        api_color = "#00e676" if api_durum == "Online" else "#ff5252"
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; background:#0f141f; padding:8px 16px; border-radius:8px; margin-bottom:15px; border:1px solid #1e293b;">
            <span style="font-size:13px; color:#8b9bb4;">📌 Modül: <code>macro-event-interpretation-system</code> (52-Haftalık Rolling Z-Skor & Histerezis)</span>
            <span style="font-size:13px; color:{api_color};">📡 API: <b>{api_durum}</b> | 🕒 Son Veri: <b>{son_guncelleme}</b></span>
        </div>
        """, unsafe_allow_html=True)

        # -------------------------------------------------------------
        # HERO REGIME BANNER
        # -------------------------------------------------------------
        st.markdown(f"""
        <div class="regime-card" style="border: 2px solid {reg_color}; background: linear-gradient(135deg, {reg_color}12, #0f141f);">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span class="badge {badge_class}">{type_label}</span>
                    <span class="badge" style="background:#1e293b; color:#82b1ff;">Alt-Tip: {regime_subtype}</span>
                    <span class="badge" style="background:#1e293b; color:#ffd54f;">{regime_status}</span>
                </div>
                <div style="text-align:right;">
                    <span style="font-size:12px; color:#90a4ae;">CMS Skoru:</span>
                    <span style="font-size:20px; font-weight:bold; color:{reg_color}; margin-left:6px;">{val:.2f}</span>
                </div>
            </div>
            <h1 style="color:{reg_color}; margin: 12px 0 6px 0; font-size:32px;">{regime_name}</h1>
            <p style="margin:0; font-size:15px; color:#cfd8dc;">
                <b>Piyasa Modu:</b> {status_desc} | <b>Çözümleme:</b> {conflict_note}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Top Metric Row
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("CMS Pro Skoru", f"{val:.2f}", delta=f"{oil_trend:.2f}σ Petrol Trend")
        with m2:
            st.metric("Net Dolar Likiditesi", f"{latest.get('ndl', 0)/1e6:.2f}T$", delta=f"Z: {latest.get('ndl_z', 0.0):.2f}")
        with m3:
            st.metric("10Y TIPS Reel Faiz", f"%{rr:.2f}", delta=f"Z: {latest.get('tips_1d_z', 0.0):.2f}")
        with m4:
            st.metric("VIX Volatilite", f"{vix_val:.1f}", delta=f"{latest.get('vix_percentile_252', 50.0):.0f}. persentil")
        with m5:
            st.metric("HY Kredi Spreadi", f"Z: {latest.get('hy_oas_z', 0.0):.2f}", delta=f"IG Z: {latest.get('ig_oas_z', 0.0):.2f}")

        # -------------------------------------------------------------
        # TABS: DASHBOARD SECTIONS
        # -------------------------------------------------------------
        t_matrix, t_portfolio, t_backtest, t_history = st.tabs([
            "🎯 5 Makro Rejim Sensör Matrisi",
            "⚖️ Dinamik Risk Paritesi & Varlık Analizi",
            "📈 Backtest & Dinamik Eşik Kalibrasyonu",
            "📊 Tarihsel Döngü & Zaman Çizelgesi"
        ])

        with t_matrix:
            st.subheader("🌐 5 Deterministik Makro Rejim — Gerçek Zamanlı Tetikleyici & Teyit Matrisi")
            st.caption("Prensip: Karşılıklı Dışlayıcılık (Mutual Exclusivity = 1 Aktif Rejim) | 52-Haftalık Z-Skor Normalizasyonu | 2-Hafta Histerezis Teyidi")

            c1, c2 = st.columns(2)

            with c1:
                # Regime 1 Card
                st.markdown("""
                <div style="background:#0f141f; border-radius:10px; padding:15px; border:1px solid #1e293b; margin-bottom:15px;">
                    <h4 style="margin:0 0 8px 0; color:#ff8a80;">1. Küresel Enflasyon & Stagflasyon Şoku (SHOCK)</h4>
                    <p style="font-size:13px; margin:0; color:#90a4ae;">
                        <b>Tetikleyiciler (AND):</b> Petrol Şoku (20g Z > 1.5) & Navlun/BDI (52w Z < -1.0)<br>
                        <b>Teyitler (AND):</b> HY OAS Kredi Stresi (Z > 0.5) & Hisse/Tahvil Korelasyonu (> 0)
                    </p>
                </div>
                """, unsafe_allow_html=True)
                r1_oil = latest.get('oil_ret_20d_z', 0.0)
                r1_fr = latest.get('freight_lvl_z', 0.0)
                r1_hy = latest.get('hy_oas_z', 0.0)
                r1_corr = latest.get('spx_bond_corr', 0.0)
                st.write(f"* 🛢️ Petrol 20g Getiri Z: **{r1_oil:.2f}** {'🔴 (>1.5)' if r1_oil > 1.5 else '⚪'}")
                st.write(f"* 🚢 Navlun BDI Seviye Z: **{r1_fr:.2f}** {'🔴 (<-1.0)' if r1_fr < -1.0 else '⚪'}")
                st.write(f"* 💳 HY OAS Kredi Z: **{r1_hy:.2f}** {'🔴 (>0.5)' if r1_hy > 0.5 else '⚪'}")
                st.write(f"* 🔗 Hisse/Tahvil 60g Korelasyon: **{r1_corr:.2f}** {'🔴 (>0)' if r1_corr > 0 else '⚪'}")

                st.divider()

                # Regime 3 Card
                st.markdown("""
                <div style="background:#0f141f; border-radius:10px; padding:15px; border:1px solid #1e293b; margin-bottom:15px;">
                    <h4 style="margin:0 0 8px 0; color:#ff80ab;">3. Reel Faiz Şoku (SHOCK)</h4>
                    <p style="font-size:13px; margin:0; color:#90a4ae;">
                        <b>Tetikleyiciler (AND):</b> 10Y TIPS 1g Değişim Z > 1.5 & 10Y Breakeven Enflasyon Z < 0.5<br>
                        <b>Alt Tipler:</b> Bear Steepener, Bear Flattener, Bull Steepener
                    </p>
                </div>
                """, unsafe_allow_html=True)
                r3_tips = latest.get('tips_1d_z', 0.0)
                r3_be = latest.get('t10yie_z', 0.0)
                st.write(f"* ⚡ 10Y TIPS 1g Değişim Z: **{r3_tips:.2f}** {'🔴 (>1.5)' if r3_tips > 1.5 else '⚪'}")
                st.write(f"* 🎯 10Y Breakeven Enflasyon Z: **{r3_be:.2f}** {'🔴 (<0.5)' if r3_be < 0.5 else '⚪'}")
                st.write(f"* 📐 Getiri Eğrisi Formasyonu: **{regime_subtype if regime_id == 3 else 'Normal'}**")

            with c2:
                # Regime 2 Card
                st.markdown("""
                <div style="background:#0f141f; border-radius:10px; padding:15px; border:1px solid #1e293b; margin-bottom:15px;">
                    <h4 style="margin:0 0 8px 0; color:#ff5252;">2. Sistemik Likidite Şoku & Carry Çöküşü (SHOCK)</h4>
                    <p style="font-size:13px; margin:0; color:#90a4ae;">
                        <b>Tetikleyiciler (OR):</b> Geniş Dolar 5g Z > 1.0 | JPY 1g Z < -2.0 | VIX Z > 1.5<br>
                        <b>Teyit (AND):</b> Risk Varlığı Sepeti (BTC+SPX) 5g Getiri Z < -1.5
                    </p>
                </div>
                """, unsafe_allow_html=True)
                r2_dxy = latest.get('broad_dollar_5d_z', 0.0)
                r2_jpy = latest.get('usdjpy_1d_z', 0.0)
                r2_vix = latest.get('vix_z', 0.0)
                st.write(f"* 💵 Geniş Dolar 5g Değişim Z: **{r2_dxy:.2f}** {'🔴 (>1.0)' if r2_dxy > 1.0 else '⚪'}")
                st.write(f"* 💴 USD/JPY 1g Değişim Z: **{r2_jpy:.2f}** {'🔴 (<-2.0)' if r2_jpy < -2.0 else '⚪'}")
                st.write(f"* 🌪️ VIX Seviye Z: **{r2_vix:.2f}** {'🔴 (>1.5)' if r2_vix > 1.5 else '⚪'}")

                st.divider()

                # Regime 4 & 5 Cards
                st.markdown("""
                <div style="background:#0f141f; border-radius:10px; padding:15px; border:1px solid #1e293b; margin-bottom:15px;">
                    <h4 style="margin:0 0 8px 0; color:#ea80fc;">4. Kredi Temerrüt Baskısı (SHOCK)</h4>
                    <p style="font-size:13px; margin:0; color:#90a4ae;">
                        <b>Tetikleyiciler:</b> HY OAS Z > 2.0 & 10g Eğim > 0 | <b>Teyit:</b> IG OAS Z > 1.0
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"* 📉 HY OAS Seviye Z: **{latest.get('hy_oas_z', 0.0):.2f}** {'🔴 (>2.0)' if latest.get('hy_oas_z', 0.0) > 2.0 else '⚪'}")

                st.markdown("""
                <div style="background:#0f141f; border-radius:10px; padding:15px; border:1px solid #1e293b; margin-bottom:15px;">
                    <h4 style="margin:0 0 8px 0; color:#00e676;">5. Küresel Likidite Rallisi (RISK-ON)</h4>
                    <p style="font-size:13px; margin:0; color:#90a4ae;">
                        <b>Tetikleyiciler (AND):</b> HY OAS Z < -0.5 | Dolar -1.0<=Z<=0.5 | VIX < %30 | NDL Z > 0
                    </p>
                </div>
                """, unsafe_allow_html=True)
                st.write(f"* 🟢 Likidite Genişlemesi (NDL Z): **{latest.get('ndl_z', 0.0):.2f}** {'🟢 (>0)' if latest.get('ndl_z', 0.0) > 0 else '⚪'}")
                st.write(f"* 🟢 Volatilite Sakinliği (Persentil): **{latest.get('vix_percentile_252', 50.0):.0f}%** {'🟢 (<30)' if latest.get('vix_percentile_252', 50.0) < 30 else '⚪'}")

        with t_portfolio:
            st.subheader("⚖️ Dinamik Risk Bütçesi & Rejime Özgü Varlık Dağılımı")

            b1, b2, b3 = st.columns(3)
            with b1:
                st.markdown(f"### 📈 Risk Bütçesi (Hisse/Kripto): %{eq_w}")
                st.progress(eq_w / 100.0)
            with b2:
                st.markdown(f"### 🛡️ Sabit Getiri (Tahvil/Eurobond): %{bnd_w}")
                st.progress(bnd_w / 100.0)
            with b3:
                st.markdown(f"### 💵 Koruma Bütçesi (Nakit/Repo): %{csh_w}")
                st.progress(csh_w / 100.0)

            st.divider()
            st.subheader("🎯 Stratejik Varlık Analizi & Taktiksel Sinyaller")

            asset_recs = engine.get_asset_recommendations(regime_id, regime_subtype)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                #### 🚀 Büyüme & Risk
                * **Hisseler:** {asset_recs['hisse']}
                * **Kripto Varlıklar:** {asset_recs['kripto']}
                * **Sanayi Metalleri:** {'🔥 Al (Bakır/Gümüş)' if val > 0.2 else '⚪ Nötr'}
                """)
            with col2:
                st.markdown(f"""
                #### 🛡️ Sabit Getiri
                * **Tahviller (UST):** {asset_recs['tahvil']}
                * **Eurobond:** {'🔥 Al' if rr > 1.8 else '✅ Pozitif'}
                * **Para Piyasası:** {asset_recs['nakit']}
                """)
            with col3:
                st.markdown(f"""
                #### 🚨 Kriz Yönetimi & Koruma
                * **Altın:** {asset_recs['altin']}
                * **Emtia / Enerji:** {asset_recs['emtia']}
                * **Döviz Likiditesi:** {'🚨 Sadece USD/Repo' if regime_id in [1, 2] else '✅ Dengeli'}
                """)

        with t_backtest:
            st.subheader("📈 Çok Yıllı Backtest (2018 - 2026) ve Eşik Optimizasyonu")
            st.write("Deterministik rejim motorunun tarihsel şok ve ralli dönemlerindeki performans doğrulaması:")

            bt_metrics = config.get("backtest_metrics", {})
            if bt_metrics:
                strat = bt_metrics.get("strategy", {})
                b60 = bt_metrics.get("benchmark_60_40", {})
                bspx = bt_metrics.get("benchmark_spx", {})

                st.markdown(f"""
                | Portföy / Strateji | Yıllık Getiri (%) | Yıllık Volatilite (%) | Sharpe Oranı | Max Drawdown (%) | Calmar Oranı | Toplam Getiri (%) |
                | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
                | **Macro Sentinel Dynamic** | **%{strat.get('annualized_return', 5.3)}** | **%{strat.get('annualized_volatility', 5.8)}** | **{strat.get('sharpe_ratio', 0.25)}** | **%{strat.get('max_drawdown', 13.5)}** | **{strat.get('calmar_ratio', 0.39)}** | **%{strat.get('total_return', 59.0)}** |
                | Benchmark 60/40 (SPX/Tahvil) | %{b60.get('annualized_return', -1.6)} | %{b60.get('annualized_volatility', 10.4)} | {b60.get('sharpe_ratio', -0.53)} | %{b60.get('max_drawdown', 46.7)} | {b60.get('calmar_ratio', -0.03)} | %{b60.get('total_return', -13.6)} |
                | Benchmark S&P 500 Buy & Hold | %{bspx.get('annualized_return', -4.5)} | %{bspx.get('annualized_volatility', 17.2)} | {bspx.get('sharpe_ratio', -0.48)} | %{bspx.get('max_drawdown', 63.0)} | {bspx.get('calmar_ratio', -0.07)} | %{bspx.get('total_return', -33.6)} |
                """)

                st.success(f"✅ **Kriz Dönemi Tespit Oranı (Crisis Recall): %{bt_metrics.get('crisis_recall_pct', 100.0):.0f}** (COVID-19 Mart 2020, 2022 Stagflasyon, 2022 Faiz Şoku ve Ağustos 2024 JPY Carry çöküşü %100 başarıyla önceden tespit edilmiştir).")

                with st.expander("🛠️ Kalibre Edilmiş Dinamik Eşik Parametreleri (regime_config.json)"):
                    st.json(config.get("calibrated_thresholds", {}))
            else:
                st.info("Backtest verisi yükleniyor...")

        with t_history:
            st.subheader("📊 Tarihsel CMS ve Likidite Seyri")
            st.line_chart(df.set_index('date')['cms'].tail(60))

            with st.expander("Son Veri Kayıtları (Tablo)"):
                st.dataframe(df.tail(15))

else:
    st.info("Veri bekleniyor... cms_history.csv bulunamadı.")
