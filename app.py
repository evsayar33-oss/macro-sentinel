import streamlit as st
import pandas as pd
import os
import numpy as np

# --- 0. TASARIM ---
st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #05070a; color: white; }
    .regime-box { padding: 25px; border-radius: 15px; border: 3px solid; text-align: center; margin-bottom: 20px; }
    .asset-card { background-color: #0f121a; padding: 20px; border-radius: 12px; border: 1px solid #30363d; height: 100%; }
    .active-strategy { border: 2px solid #00ff00; box-shadow: 0 0 15px rgba(0,255,0,0.2); }
    .not-recommended { opacity: 0.5; filter: grayscale(0.8); }
    .recommendation-text { font-size: 14px; font-weight: bold; margin-top: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ ULTIMATE MACRO SENTINEL (PRO)")

# --- 1. VERİ YÜKLEME ---
if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val = latest['cms']
        real_rate = latest.get('real_rate', 0)
        
        # REJİM TESPİTİ
        if val > 0.4: regime, col, mode = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00", "HÜCUM"
        elif 0.0 < val <= 0.4: regime, col, mode = "UZUN VADELİ KORUYUCU", "#76ff03", "STABİLİTE"
        elif -0.4 <= val <= 0.0: regime, col, mode = "SIKIŞMA / SAVUNMA", "#ffcc00", "SAVUNMA"
        else: regime, col, mode = "KRİZ / RESESYON", "#ff4b4b", "KORUMA"

        # ÜST BANNER
        st.markdown(f"""
            <div class="regime-box" style="border-color: {col}; background-color: {col}10;">
                <h1 style="color: {col}; margin: 0;">{regime}</h1>
                <h2 style="margin: 5px 0;">CMS PRO SKORU: {val:.2f}</h2>
                <p style="font-size: 18px; margin: 0;">Portföy Stratejisi: <b>{mode}</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()

        # --- 2. AKILLI VARLIK ANALİZ PANELLERİ ---
        st.subheader("🎯 Mevcut Verilere Göre Varlık Analizi")
        
        # Faiz ve Altın Dinamik Notları
        faiz_notu = "Yüksek" if real_rate > 1.5 else "Pozitif" if real_rate > 0.5 else "Düşük/Negatif"
        altin_notu = "Önerilmez (Reel Faiz Var)" if real_rate > 0.8 else "Güçlü Önerilir (Faiz Yok)"

        c1, c2, c3 = st.columns(3)

        # SENARYO 1: PARA POMPALANDIĞI ZAMAN
        with c1:
            is_active = val > 0.4
            st.markdown(f"""
            <div class="asset-card {'active-strategy' if is_active else 'not-recommended'}">
                <h3 style="color:#00ff00;">🚀 LİKİDİTE BOĞASI</h3>
                <p style="font-size:12px; color:#8b949e;">QE / Para Basımı Dönemi</p>
                <hr style="border:0.1px solid #333;">
                <ul>
                    <li>Hisseler (Hücum)</li>
                    <li>Kripto Paralar (Agresif)</li>
                    <li>Değerli Metaller (Altın, Gümüş, Bakır)</li>
                </ul>
                <p class="recommendation-text">{"👉 ŞU AN UYGUN: Agresif Alım Yapılabilir" if is_active else "⚪ Beklemede: Para girişi yetersiz"}</p>
            </div>
            """, unsafe_allow_html=True)

        # SENARYO 2: UZUN VADELİ KORUYUCU
        with c2:
            is_active = 0.0 < val <= 0.4
            st.markdown(f"""
            <div class="asset-card {'active-strategy' if is_active else 'not-recommended'}">
                <h3 style="color:#76ff03;">🛡️ UZUN VADE KORUYUCU</h3>
                <p style="font-size:12px; color:#8b949e;">Düşük Risk / İstikrar</p>
                <hr style="border:0.1px solid #333;">
                <ul>
                    <li>Gayrimenkul</li>
                    <li>Altın (Stratejik)</li>
                    <li>EuroBond</li>
                    <li>Değişken Tahviller</li>
                    <li>Yabancı Borsa Endeksleri</li>
                </ul>
                <p class="recommendation-text">{"👉 ŞU AN UYGUN: Sabit Getiri ve İstikrar" if is_active else "⚪ Beklemede: Riskler henüz dengelenmedi"}</p>
            </div>
            """, unsafe_allow_html=True)

        # SENARYO 3: KRİZ YÖNETİMİ
        with c3:
            is_active = val <= 0.0 # Sıkışma ve Kriz dönemlerini kapsar
            st.markdown(f"""
            <div class="asset-card {'active-strategy' if is_active else 'not-recommended'}">
                <h3 style="color:#ff4b4b;">🚨 KRİZ YÖNETİMİ</h3>
                <p style="font-size:12px; color:#8b949e;">Resesyon / Sıkılaşma</p>
                <hr style="border:0.1px solid #333;">
                <ul>
                    <li>Döviz Faiz (Reel Kazanç: <b>{faiz_notu}</b>)</li>
                    <li>Emtialar (Arz Kısıtlı, Seçici Ol)</li>
                    <li>Serbest Fonlar (ETF - Pozitif Akış)</li>
                    <li>Altın (Durum: <b>{altin_notu}</b>)</li>
                </ul>
                <p class="recommendation-text">{"👉 ŞU AN UYGUN: Defansif Kal, Nakit Koru" if is_active else "⚪ Beklemede: Kriz riski düşük"}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- 3. TEKNİK ÖZET ---
        st.subheader("🔍 Kurumsal Katman Analizi")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.write("### 🌐 Global Likidite (L2)")
            st.write(f"G3 Bilanço: **{latest['g3_liq']/1e6:.2f}T$**")
            st.write(f"Net Dolar Likiditesi: **{latest['ndl']/1e6:.2f}T$**")
            st.progress(min(max((val + 1) / 2, 0.0), 1.0))
        
        with m2:
            st.write("### ⚡ High-Frequency (L3)")
            st.write(f"Bakır/Altın Rasyosu: **{latest['copper_gold']:.4f}**")
            st.write(f"PMI Büyüme İvmesi: **{latest['pmi_ivme']}**")
            st.write(f"10Y Reel Faiz: **%{real_rate}**")
            
        with m3:
            st.write("### 📈 CMS Trend")
            st.line_chart(df.set_index('date')['cms'].tail(20))

    else:
        st.info("Veri bekleniyor... GitHub Actions'ı manuel tetikleyin.")
else:
    st.error("cms_history.csv dosyası bulunamadı. Lütfen ana dizinde olduğundan emin olun.")
