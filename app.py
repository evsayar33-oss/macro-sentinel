import streamlit as st
import pandas as pd
import os
import numpy as np

# --- 0. TASARIM ---
st.set_page_config(page_title="Macro Sentinel Pro", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .stMetric { background-color: #0f121a; padding: 15px; border-radius: 10px; border: 1px solid #1e222d; }
    .regime-box { padding: 30px; border-radius: 15px; border: 3px solid; text-align: center; margin-bottom: 30px; }
    .asset-group { background-color: #0f121a; padding: 20px; border-radius: 12px; border: 1px solid #30363d; height: 100%; }
    .highlight { color: #00ff00; font-weight: bold; }
    .warning { color: #ffcc00; font-weight: bold; }
    .danger { color: #ff4b4b; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ COMPOSITE MACRO GATEKEEPER")

# --- 1. VERİ KONTROLÜ ---
if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        
        # REJİM TESPİTİ
        if val > 0.5: 
            regime, col, mode = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00", "BÜYÜME"
        elif 0.1 < val <= 0.5: 
            regime, col, mode = "UZUN VADELİ KORUYUCU", "#76ff03", "STABİLİTE"
        elif -0.5 <= val <= 0.1: 
            regime, col, mode = "SIKIŞMA / NÖTR", "#ffcc00", "SAVUNMA"
        else: 
            regime, col, mode = "KRİZ / RESESYON", "#ff4b4b", "KORUMA"

        # ÜST PANEL
        st.markdown(f"""
            <div class="regime-box" style="border-color: {col}; background-color: {col}05;">
                <h1 style="color: {col}; margin: 0;">{regime}</h1>
                <h2 style="margin: 10px 0;">CMS Skoru: {val:.2f}</h2>
                <p style="font-size: 20px;">Portföy Modu: <b>{mode}</b></p>
            </div>
        """, unsafe_allow_html=True)

        # --- 2. ÜÇ ANA SENARYO VE AKSİYON PLANI ---
        st.subheader("🎯 Stratejik Varlık Dağılım Matrisi")
        
        c1, c2, c3 = st.columns(3)

        with c1:
            is_active = val > 0.5
            st.markdown(f"""
            <div class="asset-group" style="border-top: 5px solid {'#00ff00' if is_active else '#30363d'};">
                <h3>🚀 LİKİDİTE BOĞASI</h3>
                <p style="font-size: 14px; opacity: 0.8;">Para pompalandığında:</p>
                <ul>
                    <li class="{'highlight' if is_active else ''}">Hisseler (Hücum)</li>
                    <li class="{'highlight' if is_active else ''}">Kripto Paralar (Agresif)</li>
                    <li class="{'highlight' if is_active else ''}">Değerli Metaller (Altın, Gümüş, Bakır)</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF - Hücum Et' if is_active else 'Pasif'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            is_active = 0.1 < val <= 0.5
            st.markdown(f"""
            <div class="asset-group" style="border-top: 5px solid {'#76ff03' if is_active else '#30363d'};">
                <h3>🛡️ UZUN VADE KORUYUCU</h3>
                <p style="font-size: 14px; opacity: 0.8;">Düşük Risk / İstikrarlı:</p>
                <ul>
                    <li class="{'highlight' if is_active else ''}">Gayrimenkul</li>
                    <li class="{'highlight' if is_active else ''}">Altın (Stratejik)</li>
                    <li class="{'highlight' if is_active else ''}">EuroBond</li>
                    <li class="{'highlight' if is_active else ''}">Değişken Tahviller</li>
                    <li class="{'highlight' if is_active else ''}">Yabancı Borsa Endeksleri</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF - İstikrar Koru' if is_active else 'İzleme'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            is_active = val < -0.5
            st.markdown(f"""
            <div class="asset-group" style="border-top: 5px solid {'#ff4b4b' if is_active else '#30363d'};">
                <h3>🚨 KRİZ YÖNETİMİ</h3>
                <p style="font-size: 14px; opacity: 0.8;">Resesyon / Sıkılaşma:</p>
                <ul>
                    <li class="{'danger' if is_active else ''}">Döviz Faiz (Reel Kazançlı)</li>
                    <li class="{'danger' if is_active else ''}">Emtialar (Arz Kısıtlı)</li>
                    <li class="{'danger' if is_active else ''}">Serbest Fonlar (ETF - Pozitif Akış)</li>
                    <li class="{'danger' if is_active else ''}">Altın (Reel Faiz Yoksa)</li>
                </ul>
                <p><b>Durum:</b> {'TEHLİKE - Defansa Çekil' if is_active else 'Güvenli'}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- 3. TEKNİK VERİLER VE GRAFİK ---
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("📊 Makro Omurga")
            st.metric("Net Likidite (NDL)", f"{latest['ndl']/1e6:.2f}T$", delta=f"PMI İvme: {latest['pmi_ivme']}")
            st.write("---")
            st.write("**Model Notu:**")
            if val < 0:
                st.warning("Likidite yerçekimi negatifte. Riskli varlıklarda kaldıraç azaltılmalı.")
            else:
                st.success("Likidite rüzgarı arkadan esiyor. Trendi takip et.")

        with col_right:
            st.subheader("📈 CMS Döngü Takibi")
            st.line_chart(df.set_index('date')['cms'].tail(60))

    else:
        st.warning("Veri dosyası boş. Lütfen GitHub Actions'ı manuel tetikleyin.")
else:
    st.info("Sistem başlatılıyor...")
