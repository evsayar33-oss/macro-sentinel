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
        real_rate = latest.get('real_rate', 0)
        
        # REJİM TESPİTİ (Plandaki Kesin Eşikler)
        if val > 0.4: regime, col = "PİYASAYA PARA POMPALANIYOR (QE)", "#00ff00"
        elif 0.0 < val <= 0.4: regime, col = "UZUN VADELİ KORUYUCU", "#76ff03"
        elif -0.4 <= val <= 0.0: regime, col = "SIKIŞMA / NÖTR", "#ffcc00"
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

        # Otomatik Faiz Uyarıları
        faiz_durumu = "Reel Kazanç Yüksek" if real_rate > 1.8 else "Reel Kazanç Pozitif" if real_rate > 0.5 else "Reel Kazanç Düşük"
        altin_durumu = "Önerilmez (Faiz Baskısı Var)" if real_rate > 1.2 else "Seçici Ol / İzle" if real_rate > 0 else "Güçlü Önerilir"

        with c1:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:340px;">
                <h3 style="color:#00ff00;">🚀 PARA POMPALANDIĞI ZAMAN</h3>
                <p style="font-size:14px; opacity:0.8;">Hücum ve Büyüme Odaklı:</p>
                <ul>
                    <li><b>Hisseler:</b> Tam Kapasite</li>
                    <li><b>Kripto Paralar:</b> Agresif Alım</li>
                    <li><b>Metaller:</b> Altın, Gümüş, Bakır</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF' if val > 0.4 else 'Pasif'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:340px;">
                <h3 style="color:#76ff03;">🛡️ UZUN VADELİ KORUYUCU</h3>
                <p style="font-size:14px; opacity:0.8;">Düşük Risk ve İstikrar:</p>
                <ul>
                    <li>Gayrimenkul ve Eurobond</li>
                    <li>Altın (Stratejik Biriktirme)</li>
                    <li>Yabancı Borsa Endeksleri</li>
                    <li>Değişken Tahviller</li>
                </ul>
                <p><b>Durum:</b> {'AKTİF' if 0.0 < val <= 0.4 else 'İzleme'}</p>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div style="background-color:#0f121a; padding:20px; border-radius:12px; border:1px solid #30363d; height:340px;">
                <h3 style="color:#ff4b4b;">🚨 KRİZ ZAMANI</h3>
                <p style="font-size:14px; opacity:0.8;">Savunma ve Nakit:</p>
                <ul>
                    <li>Döviz Faiz (<b>{faiz_durumu}</b>)</li>
                    <li>Emtialar (Arz Kısıtlı, Seçici)</li>
                    <li>Serbest Fonlar (Pozitif Akışlı)</li>
                    <li>Altın (<b>{altin_durumu}</b>)</li>
                </ul>
                <p><b>Durum:</b> {'TEHLİKE' if val < -0.4 else 'Güvenli'}</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # --- 3. METRİKLER ---
        m1, m2, m3 = st.columns(3)
        m1.metric("Net Likidite (NDL)", f"{latest['ndl']/1e6:.2f}T$", delta="Milyon $ Bazında")
        m2.metric("PMI Büyüme İvmesi", f"{latest['pmi_ivme']}", delta="Trend Yönü")
        m3.metric("10Y Reel Faiz", f"%{real_rate}", delta="FRED Canlı")

        st.subheader("📈 CMS Döngü Takibi")
        if len(df) > 1:
            st.line_chart(df.set_index('date')['cms'].tail(60))
        else:
            st.info("Trend grafiği için bir sonraki günün verisi bekleniyor.")

    else:
        st.warning("Veri bekleniyor... GitHub Actions'ı çalıştırın.")
