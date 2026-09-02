import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Ultimate Macro Sentinel", layout="wide")
st.markdown("<style>.main { background-color: #05070a; color: white; }</style>", unsafe_allow_html=True)

if os.path.exists("cms_history.csv"):
    df = pd.read_csv("cms_history.csv")
    if not df.empty:
        latest = df.iloc[-1]
        val, rr = latest['cms'], latest['real_rate']
        
        vix_term = latest.get('vix_term', 0.85)
        ml_conf = latest.get('ml_confidence', 100)
        eq_w = latest.get('eq_weight', 50)
        bnd_w = latest.get('bond_weight', 30)
        csh_w = latest.get('cash_weight', 20)
        act_growth = latest.get('active_growth_name', 'Bakir/Altin')
        oil_trend = latest.get('oil_trend', 0.0)
        
        # REJİM BELİRLEME
        if val > 0.4: reg, col, status = "LİKİDİTE BOĞASI (QE)", "#00ff00", "HÜCUM"
        elif 0.0 < val <= 0.4: reg, col, status = "UZUN VADELİ KORUYUCU", "#76ff03", "STABİLİTE"
        elif -0.4 <= val <= 0.0: reg, col, status = "SIKIŞMA / SAVUNMA", "#ffcc00", "SAVUNMA"
        else: reg, col, status = "KRİZ / RESESYON", "#ff4b4b", "KORUMA"

        st.title("🏛️ ULTIMATE MACRO SENTINEL (AUTONOMOUS AI)")
        st.markdown(f"""
            <div style="padding:20px; border-radius:15px; border:3px solid {col}; background:{col}05; text-align:center;">
                <h1 style="color:{col}; margin:0;">{reg}</h1>
                <h2 style="margin:5px 0;">CMS PRO SKORU: {val:.2f}</h2>
                <p style="font-size:14px; opacity:0.8;">Piyasa Modu: <b>{status}</b> | Motor Ağırlıkları: {latest.get('w_str', 'Hesaplanıyor...')}</p>
            </div>
        """, unsafe_allow_html=True)

        a_notu = "Önerilmez (Faiz Baskısı)" if rr > 0.8 else "Güçlü Koruyucu"
        if oil_trend >= 1.0: emtia_notu = "🔥 Enerji/Petrol Al (Makro Teyitli)"
        elif oil_trend <= -1.0: emtia_notu = "⚠️ Zayıf (Talep Yok / Düşüş)"
        else: emtia_notu = "⚖️ Nötr (Geçici/Sahte Dalgalanma)"

        st.subheader("🎯 Stratejik Varlık Analizi")
        v1, v2, v3 = st.columns(3)
        
        with v1:
            st.markdown(f"### 🚀 Büyüme (Risk-On)\n* **Hisseler:** {'✅ Tam Kapasite' if val > 0.4 else '⚪ İzle'}\n* **Kripto:** {'🚀 Agresif Al' if val > 0.4 else '⚪ İzle'}\n* **Endüstriyel/Teknoloji:** {'🔥 Al' if val > 0.2 else '⚪ Nötr'}")
        with v2:
            st.markdown(f"### 🛡️ Sabit/Düşük Risk\n* **Gayrimenkul:** {'✅ Stabil' if val > -0.2 else '⚠️ Bekle'}\n* **Eurobond:** {'🔥 Al' if rr > 1.8 else '✅ Pozitif'}\n* **Tahviller:** {'✅ Ekle' if rr > 1.0 else '⚠️ Azalt'}\n* **Yabancı Endeksler:** {'✅ Pozitif' if val > 0 else '⚠️ Defansif'}")
        with v3:
            f_notu = "Reel Kazanç Yüksek" if rr > 1.8 else "Reel Kazanç Pozitif"
            st.markdown(f"### 🚨 Kriz Yönetimi\n* **Döviz Faiz:** ({f_notu})\n* **Emtialar (Enerji):** {emtia_notu}\n* **ETFler:** (Pozitif Akış)\n* **Altın:** ({a_notu})")

        # DİNAMİK RİSK PARİTESİ ETİKETLEME
        if val > 0.4: risk_label = "Teknoloji, Kripto, Gümüş/Bakır"
        elif val > 0.2: risk_label = "Geniş Hisseler, Endüstriyel Emtia"
        elif val > 0.0: risk_label = "Yabancı Endeksler, Seçici Hisseler"
        else: risk_label = "Sadece Defansif Hisseler (İzlemede)"

        if rr > 1.8 and val > -0.2: bond_label = "Eurobond, Tahvil, Gayrimenkul"
        elif rr > 1.0: bond_label = "Tahvil, Eurobond"
        else: bond_label = "Kısa Vadeli Tahvil / Para Piyasası"

        if rr <= 0.8: safe_label = "Nakit, Fiziki Altın (Güçlü Koruma)"
        elif oil_trend >= 1.0: safe_label = "Nakit, Repo, Enerji (Hedge)"
        else: safe_label = "Sadece Nakit / USD / Repo"

        st.divider()
        st.subheader("⚖️ Dinamik Risk Paritesi (Makro-Sensörlü)")
        
        # YENİ: Metni "Ensemble (Çoklu-Vade)" olarak güncelledik.
        if ml_conf >= 75:
            auditor_msg = f"🟢 **Ensemble Denetçi Skoru: %{ml_conf}** (Ana model onaylandı, portföy optimum.)"
        elif 40 <= ml_conf < 75:
            auditor_msg = f"🟡 **Ensemble Denetçi Skoru: %{ml_conf}** (Piyasada karmaşa var, riskli varlıklar hafif kısıldı.)"
        else:
            auditor_msg = f"🔴 **Ensemble Denetçi Skoru: %{ml_conf}** (ACİL FREN: Tüm vadelerde zayıflama var, hisse ağırlığı düşürüldü!)"
            
        st.caption(auditor_msg)
        
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"**📈 Risk Bütçesi ({risk_label}): %{eq_w}**")
            st.progress(eq_w / 100.0)
        with p2:
            st.markdown(f"**🛡️ Sabit Getiri ({bond_label}): %{bnd_w}**")
            st.progress(bnd_w / 100.0)
        with p3:
            st.markdown(f"**💵 Koruma Bütçesi ({safe_label}): %{csh_w}**")
            st.progress(csh_w / 100.0)

        st.divider()
        st.markdown("### 🌐 Bütünleşik Makro Matris (Kendi Kendini Yöneten - Auto Healing)")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            st.write("#### 💧 Likidite (L1)")
            st.write(f"G3 Bilanço: **{latest['g3_liq']/1e6:.2f}T$**")
            st.write(f"Net Dolar Likiditesi: **{latest['ndl']/1e6:.2f}T$**")
            st.write(f"10Y-2Y Eğrisi (Döngü): **{latest.get('yield_curve', 0.0)}**")
            
        with c2:
            st.write("#### ⚡ Büyüme & Risk (L2)")
            st.write(f"Sistemin Seçtiği Büyüme Endeksi: **{act_growth}**")
            st.write(f"PMI Büyüme Skoru: **{latest['pmi_z']}σ**")
            yc_status = "Genişleme" if latest.get('yield_curve', 0) > 0 else "Daralma/Resesyon"
            st.write(f"Piyasa Döngüsü: **{yc_status}**")
            
        with c3:
            st.write("#### 🧠 Sentiment & Gizli Risk (L3)")
            st.write(f"Piyasa Korkusu (VIX): **{latest['vix']}**")
            st.write(f"10Y Reel Faiz: **%{rr}**")
            vix_durum = "Normal (Contango)" if vix_term < 1.0 else "🚨 PANİK (Backwardation)"
            st.write(f"VIX Eğrisi: **{vix_term:.2f}** ({vix_durum})")
            
        st.subheader("📈 CMS Döngü Takibi (Birleştirilmiş İvme)")
        st.line_chart(df.set_index('date')['cms'].tail(30))
else:
    st.info("Veri bekleniyor...")
