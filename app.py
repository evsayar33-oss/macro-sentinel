import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🛡️ ULTIMATE MACRO SENTINEL (PRO)")

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    latest = df.iloc[-1]
    
    # REJİM MANTIĞI
    val = latest['cms']
    if val > 0.6: regime, col = "LİKİDİTE BOĞASI (QE)", "#00ff00"
    elif 0.1 < val <= 0.6: regime, col = "ERKEN GEÇİŞ / NÖTR", "#76ff03"
    elif -0.5 <= val <= 0.1: regime, col = "SIKIŞMA / SAVUNMA", "#ffcc00"
    else: regime, col = "MAKRO ÇÖKÜŞ / RESESYON", "#ff4b4b"

    # 1. ANALİST ÖZETİ (BANNER)
    st.markdown(f"""
        <div style="padding:25px; border-radius:15px; border:4px solid {col}; background:{col}05; text-align:center; margin-bottom:20px;">
            <h1 style="color:{col}; margin:0;">{regime}</h1>
            <h2 style="margin:5px 0;">CMS PRO SKORU: {val:.2f}</h2>
            <p style="font-size:16px;">Sistem Ağırlıkları (L/G/S/R): {latest['weights']}</p>
        </div>
    """, unsafe_allow_html=True)

    # 2. KATMANLI ANALİZ PANELLERİ
    st.subheader("🔍 Kurumsal Katman Analizi")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🌐 Global Likidite (L2)")
        st.write(f"G3 Bilanço Toplamı: **{latest['g3_liq']/1e6:.2f}T$**")
        st.write(f"Net Dolar Likiditesi: **{latest['ndl']/1e6:.2f}T$**")
        st.progress(min(max((val + 1) / 2, 0.0), 1.0))

    with c2:
        st.markdown("### ⚡ High-Frequency (L3)")
        st.write(f"Bakır/Altın Rasyosu: **{latest['copper_gold']:.4f}**")
        st.caption("Yükseliş = Küresel Büyüme ve Risk İştahı")
        st.write(f"Dinamik Ağırlık Katsayısı: **%{float(latest['weights'].split(',')[1])*100:.0f}**")

    with c3:
        st.markdown("### 🧠 Sentiment / Opsiyon (L5)")
        pc = latest['pc_ratio']
        st.write(f"Put/Call Oranı: **{pc}**")
        if pc > 1.0: st.error("Aşırı Korku Hakim")
        elif pc < 0.6: st.success("Aşırı Coşku / FOMO")
        else: st.info("Dengeli Beklenti")

    st.divider()

    # 3. YATIRIM REHBERİ
    st.subheader("🎯 Stratejik Varlık Konumlandırma")
    v1, v2, v3 = st.columns(3)
    
    with v1:
        st.info("🏠 **Düşük Risk (Savunma)**")
        if val < 0.1: st.write("✅ Eurobond & Altın \n✅ Gayrimenkul \n✅ Sabit Tahviller")
        else: st.write("⚪ Kademeli Azalt")

    with v2:
        st.success("🚀 **Hücum (Büyüme)**")
        if val > 0.4: st.write("✅ Nasdaq (NDX) & S&P \n✅ Kripto (BTC/ETH) \n✅ Bakır & Gümüş")
        else: st.write("⚪ Bekle / Nakit Koru")

    with v3:
        st.warning("🚨 **Kriz Zamanı**")
        if val < -0.5: st.write("✅ Döviz Faiz \n✅ Volatilite Fonları \n✅ Fiziki Altın")
        else: st.write("✅ Güvenli Bölge")

    st.subheader("📈 CMS Döngüsel Trend")
    st.line_chart(df.set_index('date')['cms'].tail(60))

else:
    st.info("Veri bekleniyor... GitHub Actions manuel tetikleyin.")
