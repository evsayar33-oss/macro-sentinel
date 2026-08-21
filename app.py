import streamlit as st
import pandas as pd
import os
import numpy as np

# --- 0. TASARIM ---
st.set_page_config(page_title="Macro Sentinel Dashboard", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🏛️ COMPOSITE MACRO GATEKEEPER")

# --- 1. VERİ KONTROLÜ ---
if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        
        # Rejim Tespiti (Ana Plana Göre)
        if val > 0.5: regime, col, risk_status = "QE BOĞA", "#00ff00", "BÜYÜME"
        elif -0.2 < val <= 0.5: regime, col, risk_status = "ERKEN GEÇİŞ", "#76ff03", "YÜKSELİŞ"
        elif -0.8 <= val <= -0.2: regime, col, risk_status = "SIKIŞMA / NÖTR", "#ffcc00", "SAVUNMA"
        else: regime, col, risk_status = "RESESYON / SIKILAŞMA", "#ff4b4b", "KORUMA"

        # ÜST PANEL
        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:3px solid {col}; background:{col}05; text-align:center;">
                <h1 style="color:{col}; margin:0;">{regime} MODU AKTİF</h1>
                <h2 style="margin:10px 0;">CMS Skoru: {val:.2f}</h2>
                <p style="font-size:20px; opacity:0.9;">Mevcut Piyasa Modu: <b>{risk_status}</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- 2. STRATEJİK TAHSİSAT (PORTFÖY REHBERİ) ---
        st.subheader("💼 Stratejik Tahsisat Rehberi")
        
        # Rejime göre ağırlıklar
        alloc_data = {
            "QE BOĞA": {"Riskli (BTC/NDX)": "80%", "Hisse (SPX)": "15%", "Nakit/Tahvil": "5%"},
            "ERKEN GEÇİŞ": {"Riskli (BTC/ETH)": "50%", "Hisse (SPX)": "30%", "Nakit/Tahvil": "20%"},
            "SIKIŞMA / NÖTR": {"Riskli (Kripto)": "10%", "Defansif (Altın/Gümüş)": "40%", "Nakit": "50%"},
            "RESESYON / SIKILAŞMA": {"Riskli": "0%", "Defansif (Altın)": "20%", "Nakit (Dolar)": "80%"}
        }.get(regime)

        cols = st.columns(len(alloc_data))
        for i, (asset_class, weight) in enumerate(alloc_data.items()):
            cols[i].metric(asset_class, weight)

        st.divider()

        # --- 3. VARLIK BAZLI AKSİYON PLANI ---
        st.subheader("🎯 6 Hedef Varlık İçin Aksiyon Planı")
        
        v1, v2, v3 = st.columns(3)
        
        # Kripto (BTC & ETH)
        with v1:
            st.markdown("### ₿ Kripto")
            if val > 0.3: st.success("**BTC & ETH:** Güçlü Al / Biriktir")
            elif val > -0.3: st.warning("**BTC & ETH:** Bekle / Sadece Spot Tut")
            else: st.error("**BTC & ETH:** Nakde Geç / Kısa Vade İzleme")

        # Endeksler (SPX & NDX)
        with v2:
            st.markdown("### 📈 Endeksler")
            if val > 0.1: st.success("**SPX & NDX:** Trend Takibi (Long)")
            elif val > -0.5: st.warning("**SPX & NDX:** Kar Realize Et / Yatay")
            else: st.error("**SPX & NDX:** Ayı Piyasası / Savunma")

        # Emtialar (XAU & XAG)
        with v3:
            st.markdown("### 🪙 Metaller")
            if val < 0.2: st.success("**ALTIN & GÜMÜŞ:** Güvenli Liman Alımı")
            else: st.warning("**ALTIN & GÜMÜŞ:** Nötr / İzle")

        st.divider()
        
        # TEKNİK METRİKLER
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Net Likidite (NDL)", f"{latest['ndl']/1e6:.2f}T$", delta=f"İvme: {latest['pmi_ivme']}")
            st.caption("Fed Bilanço - TGA - RRP (Gerçek Likidite)")
        with c2:
            st.subheader("📈 CMS Zaman Serisi")
            st.line_chart(df.set_index('date')['cms'].tail(30))
    else:
        st.info("Veri bekleniyor... GitHub Actions çalışınca tablo dolacaktır.")
