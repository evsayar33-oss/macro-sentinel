import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🛡️ ULTIMATE MACRO SENTINEL (PRO)")

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        real_rate = latest['real_rate']
        
        # REJİM TESPİTİ
        if val > 0.4: regime, col = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00"
        elif 0.0 < val <= 0.4: regime, col = "UZUN VADELİ KORUYUCU", "#76ff03"
        elif -0.4 <= val <= 0.0: regime, col = "SIKIŞMA / SAVUNMA", "#ffcc00"
        else: regime, col = "KRİZ / RESESYON", "#ff4b4b"

        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:3px solid {col}; background-color: {col}10; text-align:center;">
                <h1 style="color: {col}; margin: 0;">{regime}</h1>
                <h2 style="margin: 10px 0;">CMS PRO SKORU: {val:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- DİNAMİK ANALİZ NOTLARI ---
        faiz_notu = "Yüksek (Dövizde Kal)" if real_rate > 1.5 else "Pozitif"
        altin_notu = "Önerilmez (Reel Faiz Baskısı)" if real_rate > 1.0 else "Stratejik Biriktir"

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div style="background:#0f121a; padding:20px; border-radius:12px; border:1px solid #333; height:340px;">
                <h3 style="color:#00ff00;">🚀 LİKİDİTE BOĞASI</h3>
                <ul><li>Hisseler (Hücum)</li><li>Kripto (Agresif)</li><li>Bakır & Gümüş</li></ul>
                <p><b>Durum:</b> {'UYGUN' if val > 0.4 else 'Beklemede'}</p></div>""", unsafe_allow_html=True)

        with c2:
            st.markdown(f"""<div style="background:#0f121a; padding:20px; border-radius:12px; border:1px solid #333; height:340px;">
                <h3 style="color:#76ff03;">🛡️ UZUN VADE KORUYUCU</h3>
                <ul><li>Gayrimenkul</li><li>EuroBond</li><li>Yabancı Endeksler</li></ul>
                <p><b>Durum:</b> {'UYGUN' if 0.0 < val <= 0.4 else 'İzleme'}</p></div>""", unsafe_allow_html=True)

        with c3:
            st.markdown(f"""<div style="background:#0f121a; padding:20px; border-radius:12px; border:1px solid #333; height:340px;">
                <h3 style="color:#ff4b4b;">🚨 KRİZ YÖNETİMİ</h3>
                <ul><li>Döviz Faiz: <b>{faiz_notu}</b></li><li>Emtialar: <b>Seçici Ol</b></li><li>Altın: <b>{altin_notu}</b></li></ul>
                <p><b>Durum:</b> {'TEHLİKE: Defans' if val < -0.4 else 'Güvenli'}</p></div>""", unsafe_allow_html=True)

        st.divider()

        # --- KURUMSAL ANALİZ ---
        m1, m2, m3 = st.columns(3)
        m1.metric("G3 Küresel Bilanço", f"{latest['g3_liq']/1e6:.2f}T$", delta="Kur Düzeltmeli")
        m2.metric("PMI İvmesi", f"{latest['pmi_ivme']}", delta="Büyüme Trendi")
        m3.metric("10Y Reel Faiz", f"%{real_rate}", delta="Altın Baskısı")

        st.subheader("📈 CMS Döngüsel Trend")
        st.line_chart(df.set_index('date')['cms'].tail(30))
