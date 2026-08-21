import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Macro Sentinel Pro", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🏛️ COMPOSITE MACRO GATEKEEPER")

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        
        if val > 0.6: reg, col, desc = "QE BOĞA (RİSK-ON)", "#00ff00", "Likidite artışı ve büyüme pozitif."
        elif 0.1 < val <= 0.6: reg, col, desc = "ERKEN GEÇİŞ", "#76ff03", "Piyasa dengeleniyor, risk alınabilir."
        elif -0.5 <= val <= 0.1: reg, col, desc = "SIKIŞMA / NÖTR", "#ffcc00", "Dikkatli olunmalı, nakit değerlidir."
        else: reg, col, desc = "RESESYON / SIKILAŞMA", "#ff4b4b", "Sermaye koruma modu aktif edilmeli."

        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:3px solid {col}; background:{col}05; text-align:center;">
                <h1 style="color:{col}; margin:0;">{reg}</h1>
                <h2 style="margin:10px 0;">CMS Skoru: {val:.2f}</h2>
                <p style="font-size:18px; opacity:0.8;">{desc}</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Net Likidite (NDL)", f"{latest['ndl']/1e6:.2f}T$")
            st.metric("PMI İvmesi", f"{latest['pmi_ivme']:.2f}")
        with c2:
            st.subheader("📈 CMS Döngü Grafiği")
            st.line_chart(df.set_index('date')['cms'].tail(30))
    else:
        st.info("Veri bekleniyor... GitHub Actions üzerinden 'Run workflow' yapın.")
else:
    st.error("cms_history.csv bulunamadı!")
