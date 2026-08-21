import streamlit as st
import pandas as pd
import os

# --- 0. TASARIM ---
st.set_page_config(page_title="Macro Sentinel Pro", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🏛️ COMPOSITE MACRO GATEKEEPER")

# --- 1. VERİ KONTROLÜ ---
if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        real_rate = latest.get('real_rate', 0) # FRED'den gelen reel faiz
        
        # REJİM TESPİTİ
        if val > 0.5: regime, col = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00"
        elif 0.1 < val <= 0.5: regime, col = "UZUN VADELİ KORUYUCU", "#76ff03"
        elif -0.5 <= val <= 0.1: regime, col = "SIKIŞMA / NÖTR", "#ffcc00"
        else: regime, col = "KRİZ / RESESYON", "#ff4b4b"

        # ÜST PANEL
        st.markdown(f"""
            <div style="padding:25px; border-radius:15px; border:3px solid {col}; background-color: {col}05; text-align:center;">
                <h1 style="color: {col}; margin: 0;">{regime}</h1>
                <h2 style="margin: 10px 0;">CMS Skoru: {val:.2f}</h2>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- 2. DİNAMİK VARLIK ANALİZİ ---
        st.subheader("🎯 Stratejik Varlık Dağılım Matrisi")
        c1, c2, c3 = st.columns(3)

        # FAİZ UYARILARI MANTIĞI
        faiz_notu = "Yüksek" if real_rate > 1.5 else "Pozitif" if real_rate > 0 else "Yok/Negatif"
        altin_notu = "Önerilmez (Faiz Baskısı)" if real_rate > 1.0 else "Seçici Ol" if real_rate > 0 else "Güçlü Koruyucu"

        with c1:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:320px;">
                <h3>🚀 LİKİDİTE BOĞASI</h3>
                <ul>
                    <li>Hisseler (Hücum)</li>
                    <li>Kripto Paralar (Agresif)</li>
                    <li>Değerli Metaller (Altın, Gümüş, Bakır)</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF' if val > 0.5 else 'Pasif'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:320px;">
                <h3>🛡️ UZUN VADE KORUYUCU</h3>
                <ul>
                    <li>Gayrimenkul</li>
                    <li>Altın (Stratejik)</li>
                    <li>EuroBond</li>
                    <li>Değişken Tahviller</li>
                    <li>Yabancı Borsa Endeksleri</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF' if 0.1 < val <= 0.5 else 'İzleme'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:320px;">
                <h3>🚨 KRİZ YÖNETİMİ</h3>
                <ul>
                    <li>Döviz Faiz (Reel Kazanç: <b>{faiz_notu}</b>)</li>
                    <li>Emtialar (Arz Kısıtlı, Seçici Ol)</li>
                    <li>Serbest Fonlar (ETF - Pozitif Akış)</li>
                    <li>Altın (Reel Faiz: <b>{altin_notu}</b>)</li>
                </ul>
                <p><b>Durum:</b> {'TEHLİKE' if val < -0.5 else 'Güvenli'}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- 3. METRİKLER ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Net Likidite (NDL)", f"{latest['ndl']/1e6:.2f}T$")
        
        # PMI İvmesi Görünürlüğü Düzeltildi
        pmi_val = latest['pmi_ivme']
        pmi_col = "normal" if pmi_val == 0 else "inverse" # Renk yönetimi
        m2.metric("PMI Büyüme İvmesi", f"{pmi_val}", delta=pmi_val, delta_color=pmi_col)
        
        m3.metric("10Y Reel Faiz", f"%{real_rate}")

        st.subheader("📈 CMS Döngü Takibi")
        st.line_chart(df.set_index('date')['cms'].tail(60))

    else:
        st.warning("Veri bekleniyor... GitHub Actions'ı manuel tetikleyin.")
