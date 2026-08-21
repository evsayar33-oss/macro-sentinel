import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Ultimate Macro Sentinel Pro", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    latest = df.iloc[-1]
    val, rr = latest['cms'], latest['real_rate']
    
    # REJİM BELİRLEME
    if val > 0.4: reg, col, status = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00", "HÜCUM"
    elif 0.0 < val <= 0.4: reg, col, status = "UZUN VADELİ KORUYUCU", "#76ff03", "STABİLİTE"
    elif -0.4 <= val <= 0.0: reg, col, status = "SIKIŞMA / SAVUNMA", "#ffcc00", "SAVUNMA"
    else: reg, col, status = "KRİZ / RESESYON", "#ff4b4b", "KORUMA"

    st.title("🏛️ ULTIMATE MACRO SENTINEL (PRO)")
    st.markdown(f"""
        <div style="padding:20px; border-radius:15px; border:3px solid {col}; background:{col}05; text-align:center;">
            <h1 style="color:{col}; margin:0;">{reg}</h1>
            <h2 style="margin:5px 0;">CMS PRO SKORU: {val:.2f}</h2>
            <p style="font-size:14px; opacity:0.8;">Piyasa Modu: <b>{status}</b> | Faktör Hakimiyeti (L/G/S/R): {latest.get('weights_json', 'N/A')}</p>
        </div>
    """, unsafe_allow_html=True)

    # --- AKILLI VARLIK ANALİZ MATRİSİ ---
    st.subheader("🎯 Stratejik Varlık Analizi")
    v1, v2, v3 = st.columns(3)
    
    with v1:
        st.markdown(f"### 🚀 Büyüme (Risk-On)\n"
                    f"*   **Hisseler:** {'✅ Tam Kapasite' if val > 0.4 else '⚪ İzle / Bekle'}\n"
                    f"*   **Kripto Paralar:** {'🚀 Agresif Alım' if val > 0.4 else '⚪ Spot Tut / İzle'}\n"
                    f"*   **Değerli Metaller:** {'🔥 Uygun' if val > 0.2 else '⚪ Nötr'}")
    
    with v2:
        st.markdown(f"### 🛡️ Uzun Vade / Düşük Risk\n"
                    f"*   **Gayrimenkul:** {'✅ Stabil Trend' if val > -0.2 else '⚠️ Bekle / Riskli'}\n"
                    f"*   **Eurobond:** {'🔥 Alıma Uygun' if rr > 1.8 else '✅ Pozitif'}\n"
                    f"*   **Sabit Tahviller:** {'✅ Portföye Ekle' if rr > 1.0 else '⚠️ Azalt'}\n"
                    f"*   **Yabancı Endeksler:** {'✅ Pozitif' if val > 0 else '⚠️ Defansif'}")

    with v3:
        f_notu = "Reel Kazanç: Yüksek" if rr > 1.8 else "Reel Kazanç: Pozitif"
        a_notu = "Önerilmez (Faiz Baskısı)" if rr > 0.8 else "Güçlü Koruyucu"
        st.markdown(f"### 🚨 Kriz Yönetimi\n"
                    f"*   **Döviz Faiz:** ({f_notu})\n"
                    f"*   **Emtialar:** (Arz Kısıtlı / Seçici Ol)\n"
                    f"*   **Serbest Fonlar:** (ETF - Pozitif Akış)\n"
                    f"*   **Altın:** ({a_notu})")

    st.divider()

    # --- KURUMSAL KATMANLAR ---
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("#### 🌐 Global Likidite (L2)")
        st.write(f"G3 Bilanço Toplamı: **{latest['g3_liq']/1e6:.2f}T$**")
        st.write(f"Net Dolar Likiditesi: **{latest['ndl']/1e6:.2f}T$**")
        st.progress(min(max((val + 1) / 2, 0.0), 1.0))
    with c2:
        st.write("#### ⚡ High-Freq & Growth (L3)")
        st.write(f"Bakır/Altın Rasyosu: **{latest['copper_gold']:.4f}**")
        st.write(f"PMI Büyüme: **{latest.get('pmi_z', 0)}σ**")
        st.caption("PMI Z > 0 ise büyüme tarihsel ortalamanın üstündedir.")
    with c3:
        st.write("#### 🧠 Sentiment & Stres (L5)")
        st.write(f"Piyasa Korkusu (VIX): **{latest['vix']}**")
        st.write(f"10Y Reel Faiz: **%{rr}**")

    st.subheader("📈 CMS Döngü Takibi")
    st.line_chart(df.set_index('date')['cms'].tail(60))
else:
    st.info("Veri bekleniyor...")
