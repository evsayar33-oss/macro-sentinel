import streamlit as st
import pandas as pd
import os
import numpy as np

st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🛡️ ULTIMATE MACRO SENTINEL (PRO)")

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        
        # NaN Koruması
        val = latest['cms']
        if pd.isna(val): val = 0.0
        
        g3_liq = latest.get('g3_liq', 0)
        ndl = latest.get('ndl', 0)
        weights = latest.get('weights', "0.25,0.25,0.25,0.25")
        
        # REJİM MANTIĞI
        if val > 0.4: regime, col = "LİKİDİTE BOĞASI (QE)", "#00ff00"
        elif 0.0 < val <= 0.4: regime, col = "ERKEN GEÇİŞ / NÖTR", "#76ff03"
        elif -0.4 <= val <= 0.0: regime, col = "SIKIŞMA / SAVUNMA", "#ffcc00"
        else: regime, col = "MAKRO ÇÖKÜŞ / RESESYON", "#ff4b4b"

        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:4px solid {col}; background:{col}05; text-align:center; margin-bottom:20px;">
                <h1 style="color:{col}; margin:0;">{regime}</h1>
                <h2 style="margin:5px 0;">CMS PRO SKORU: {val:.2f}</h2>
                <p style="font-size:14px; opacity:0.7;">Dinamik Faktör Ağırlıkları: {weights}</p>
            </div>
        """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("### 🌐 Global Likidite (L2)")
            st.write(f"G3 Bilanço: **{g3_liq/1e6:.2f}T$**")
            st.write(f"Net Dolar Likiditesi: **{ndl/1e6:.2f}T$**")
            # Progress bar NaN korumalı
            progress_val = min(max((val + 1) / 2, 0.0), 1.0)
            st.progress(progress_val)

        with c2:
            st.markdown("### ⚡ High-Frequency (L3)")
            st.write(f"Bakır/Altın: **{latest.get('copper_gold', 0):.4f}**")
            st.write(f"Reel Faiz: **%{latest.get('real_rate', 0)}**")

        with c3:
            st.markdown("### 🧠 Sentiment (L5)")
            pc = latest.get('pc_ratio', 0.7)
            st.write(f"Put/Call Oranı: **{pc}**")
            if pc > 1.0: st.error("Korku Hakim")
            elif pc < 0.6: st.success("Aşırı Coşku")
            else: st.info("Dengeli")

        st.divider()
        st.subheader("📈 CMS Döngüsel Trend")
        st.line_chart(df.set_index('date')['cms'].tail(60))
    else:
        st.info("Veri bekleniyor... GitHub Actions manuel tetikleyin.")
