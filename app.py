import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    latest = df.iloc[-1]
    val, rr = latest['cms'], latest['real_rate']
    
    if val > 0.4: reg, col = "LİKİDİTE BOĞASI (QE)", "#00ff00"
    elif 0.0 < val <= 0.4: reg, col = "ERKEN GEÇİŞ / NÖTR", "#76ff03"
    elif -0.4 <= val <= 0.0: reg, col = "SIKIŞMA / SAVUNMA", "#ffcc00"
    else: reg, col = "KRİZ / RESESYON", "#ff4b4b"

    st.title("🛡️ ULTIMATE MACRO SENTINEL (PRO)")
    st.markdown(f'<div style="padding:20px; border-radius:15px; border:3px solid {col}; background:{col}05; text-align:center;">'
                f'<h1 style="color:{col}; margin:0;">{reg}</h1>'
                f'<h2 style="margin:5px 0;">CMS PRO SKORU: {val:.2f}</h2>'
                f'<p style="font-size:14px; opacity:0.7;">Anlık Faktör Ağırlıkları (L/G/S/R): {latest["w_str"]}</p></div>', unsafe_allow_html=True)

    st.subheader("🎯 Stratejik Varlık Analizi & Notlar")
    v1, v2, v3 = st.columns(3)
    
    with v1:
        st.markdown(f"### 🚀 Büyüme (Hücum)\n* **Hisseler:** {'✅ Uygun' if val > 0.4 else '⚪ Bekle'}\n* **Kripto:** {'🚀 Agresif Al' if val > 0.4 else '⚪ İzle'}\n* **Bakır/Gümüş:** {'🔥 Al' if val > 0.2 else '⚪ Nötr'}")
    with v2:
        st.markdown(f"### 🛡️ Sabit/Düşük Risk\n* **Gayrimenkul:** {'✅ Stabil' if val > -0.2 else '⚠️ Bekle'}\n* **Eurobond:** {'✅ Uygun' if rr > 1.5 else '⚪ İzle'}\n* **Yabancı Endeksler:** {'✅ Pozitif' if val > 0 else '⚠️ Defansif'}")
    with v3:
        f_notu = "Yüksek Kazanç" if rr > 1.8 else "Düşük Kazanç"
        a_notu = "Önerilmez (Faiz Baskısı)" if rr > 0.8 else "Güçlü Koruyucu"
        st.markdown(f"### 🚨 Kriz Yönetimi\n* **Döviz Faiz:** ({f_notu})\n* **Emtialar:** (Seçici Ol)\n* **ETFler:** (Pozitif Akış)\n* **Altın:** ({a_notu})")

    st.divider()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("#### 🌐 Global Likidite (L2)")
        st.write(f"G3 Bilanço: **{latest['g3_liq']/1e6:.2f}T$**")
        st.write(f"Net Dolar Likiditesi: **{latest['ndl']/1e6:.2f}T$**")
        st.progress(min(max((val + 1) / 2, 0.0), 1.0))
    with c2:
        st.write("#### ⚡ High-Freq & Growth (L3)")
        st.write(f"Bakır/Altın Rasyosu: **{latest['copper_gold']:.4f}**")
        st.write(f"PMI Büyüme: **{latest['pmi_z']}σ**")
    with c3:
        st.write("#### 🧠 Sentiment (L5)")
        st.write(f"Piyasa Korkusu (VIX): **{latest['vix']}**")
        if latest['vix'] > 25: st.error("Yüksek Korku")
        elif latest['vix'] < 15: st.success("Düşük Risk Algısı")
        else: st.info("Dengeli Sentiment")

    st.subheader("📈 CMS Döngü Takibi")
    st.line_chart(df.set_index('date')['cms'].tail(30))
else:
    st.info("Veri yükleniyor...")
