import streamlit as st
import pandas as pd
import os

# --- 0. TASARIM ---
st.set_page_config(page_title="Macro Sentinel Dashboard", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

st.title("🏛️ COMPOSITE MACRO GATEKEEPER")

# --- 1. VERİ KONTROLÜ ---
if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    
    if not df.empty:
        latest = df.iloc[-1]
        
        # Rejim Tespiti
        val = latest['cms']
        if val > 0.5: regime, col, desc = "QE BOĞA", "#00ff00", "Tam Risk-On"
        elif -0.2 < val <= 0.5: regime, col, desc = "ERKEN GEÇİŞ", "#76ff03", "Yatırımlar Artırılabilir"
        elif -0.8 <= val <= -0.2: regime, col, desc = "SIKIŞMA / NÖTR", "#ffcc00", "Savunma Modu: Nakit Ağırlıklı"
        else: regime, col, desc = "RESESYON / SIKILAŞMA", "#ff4b4b", "Risk-Off: Sermaye Koruma"

        st.markdown(f"""
            <div style="padding:20px; border-radius:15px; border:2px solid {col}; background:{col}10; text-align:center;">
                <h1 style="color:{col};">{regime}</h1>
                <h3>CMS Skoru: {val}</h3>
                <p>{desc}</p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📊 Makro Momentumlar")
            st.write(f"Net Likidite: **{latest['ndl']/1e6:.2f}T$**")
            st.write(f"PMI Büyüme İvmesi: **{latest['pmi_ivme']:.2f}**")
        
        with c2:
            st.subheader("📈 CMS Geçmişi (Trend)")
            if len(df) > 1:
                st.line_chart(df.set_index('date')['cms'].tail(30))
            else:
                st.info("Trend grafiği için daha fazla veri birikmesi gerekiyor.")
    else:
        st.warning("⚠️ Veri dosyası boş. Lütfen GitHub Actions sekmesinden 'Run workflow' butonuna basarak ilk taramayı başlatın.")
else:
    st.info("Veri bekleniyor... GitHub Actions ilk taramayı yapınca tablo dolacaktır.")
