import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Macro Sentinel v2.0",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

HISTORY_FILE = "cms_history.csv"
CONFIG_FILE = "regime_config.json"
STRESS_FILE = "stress_test_results.json"

st.markdown(
    """
    <style>
    .main { background-color:#07090e; color:#e1e7ec; }
    .stMetric { background-color:#0f141f; padding:12px; border-radius:8px; border:1px solid #1e293b; }
    .card { padding:18px; border-radius:12px; border:1px solid #1e293b; background:#0f141f; margin-bottom:14px; }
    </style>
    """,
    unsafe_allow_html=True,
)


def fnum(row, key, default=0.0):
    try:
        x = float(row.get(key, default))
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def bval(row, key, default=False):
    return str(row.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}


if not os.path.exists(HISTORY_FILE):
    st.warning("cms_history.csv bulunamadı. GitHub Actions'ın ilk başarılı veri güncellemesini bekleyin.")
    st.stop()

try:
    df = pd.read_csv(HISTORY_FILE)
except Exception as exc:
    st.error(f"Geçmiş veri okunamadı: {exc}")
    st.stop()

if df.empty:
    st.info("Henüz kayıt yok.")
    st.stop()

latest = df.iloc[-1]
health = str(latest.get("data_health_status", "UNKNOWN"))
decision = str(latest.get("decision_status", "UNKNOWN"))
source = str(latest.get("allocation_source", "UNKNOWN"))
regime_id = int(fnum(latest, "regime_id", 0))
regime_name = str(latest.get("regime_name", "REJIMSIZ_GECIS"))
regime_type = str(latest.get("regime_type", "TRANSITION"))
regime_subtype = str(latest.get("regime_subtype", "Dengeli / Nötr Piyasa"))
regime_status = str(latest.get("regime_status", ""))

weights = {
    "Hisse": fnum(latest, "eq_weight"),
    "Tahvil": fnum(latest, "bond_weight"),
    "Nakit": fnum(latest, "cash_weight"),
    "Altın": fnum(latest, "gold_weight"),
    "Emtia": fnum(latest, "commodity_weight"),
    "Kripto": fnum(latest, "crypto_weight"),
}
allocation_total = sum(weights.values())

oil_score = fnum(latest, "oil_event_score")
oil_active = bval(latest, "commodity_event_active")
unknown_score = fnum(latest, "unknown_event_score")
unknown_active = bval(latest, "unknown_event_active")

st.title("🏛️ Macro Sentinel v2.0")
st.caption("Point-in-time makro rejim + bağımsız olay sensörleri + fail-closed allocation")

health_badge = "🟢 HEALTHY" if health == "HEALTHY" else "🟡 DEGRADED" if health == "DEGRADED" else "🔴 HALT"
st.markdown(
    f"**Veri Sağlığı:** {health_badge}  |  **Karar:** `{decision}`  |  **Kaynak:** `{source}`  |  **Son Güncelleme:** `{latest.get('date', 'N/A')}`"
)

if health == "HALT":
    st.error(
        "Canlı karar motoru veri yetersizliği nedeniyle yeni allocation üretmedi. "
        "Son geçerli allocation korunuyor veya ilk çalışmada %100 nakit kullanılıyor."
    )
    if str(latest.get("data_health_issues", "")):
        st.write("**Sorunlar:**", latest.get("data_health_issues"))
elif health == "DEGRADED":
    st.warning(str(latest.get("data_health_warnings", "Bazı ikincil sensörler eksik.")))

hero_color = "#ff5252" if regime_type == "SHOCK" else "#00e676" if regime_type == "RISK_ON" else "#ffd600"
st.markdown(
    f"""
    <div class="card" style="border:2px solid {hero_color};">
      <div style="font-size:13px;color:#90a4ae;">CONFIRMED REGIME</div>
      <h2 style="margin:6px 0;color:{hero_color};">{regime_name}</h2>
      <div style="color:#cfd8dc;">ID {regime_id} · {regime_type} · {regime_subtype}</div>
      <div style="margin-top:8px;color:#90a4ae;">{regime_status}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    st.metric("CMS", f"{fnum(latest, 'cms'):.2f}")
with m2:
    st.metric("Petrol Event", f"{oil_score:.2f}")
with m3:
    st.metric("Petrol 20g Z", f"{fnum(latest, 'oil_ret_20d_z'):.2f}")
with m4:
    st.metric("VIX Z", f"{fnum(latest, 'vix_z'):.2f}")
with m5:
    st.metric("NDL Z", f"{fnum(latest, 'ndl_z'):.2f}")

if oil_active:
    st.warning(
        f"🛢️ Bağımsız petrol/emtia olayı aktif. Petrol Event={oil_score:.2f}; "
        f"Emtia allocation=%{weights['Emtia']:.2f}. "
        f"{latest.get('commodity_event_reason', '')}"
    )
if unknown_active:
    st.warning(f"⚠️ Tanımlanmamış anomali koruması aktif. Skor={unknown_score:.2f}; risk azaltıcı guard uygulanabilir.")

st.subheader("⚖️ Mevcut Allocation")
cols = st.columns(6)
for col, (name, value) in zip(cols, weights.items()):
    with col:
        st.metric(name, f"%{value:.2f}")

if abs(allocation_total - 100.0) > 0.05:
    st.error(f"Allocation toplamı %{allocation_total:.2f}. Bu bir bütünlük hatasıdır.")
else:
    st.success(f"Allocation bütünlüğü: %{allocation_total:.2f}")

with st.expander("🔍 Olay ve Sensör Detayları", expanded=True):
    details = {
        "Petrol 5g Getiri Z": fnum(latest, "oil_ret_5d_z"),
        "Petrol 20g Getiri Z": fnum(latest, "oil_ret_20d_z"),
        "Petrol 5g Anomali Persentili": fnum(latest, "oil_abs_5d_percentile"),
        "Petrol Event Kalitesi (60g)": fnum(latest, "oil_event_quality_60d", 0.5),
        "Emtia Genişliği (20g)": fnum(latest, "commodity_breadth_20d"),
        "Navlun Z": fnum(latest, "freight_lvl_z"),
        "HY OAS Z": fnum(latest, "hy_oas_z"),
        "Hisse/Tahvil Korelasyonu": fnum(latest, "spx_bond_corr"),
        "DXY 5g Z": fnum(latest, "broad_dollar_5d_z"),
        "USDJPY 1g Z": fnum(latest, "usdjpy_1d_z"),
        "10Y TIPS 1g Z": fnum(latest, "tips_1d_z"),
        "T10YIE Z": fnum(latest, "t10yie_z"),
        "Unknown Anomaly Score": unknown_score,
    }
    st.dataframe(pd.DataFrame([details]), use_container_width=True, hide_index=True)

with st.expander("🛡️ Veri Sağlığı ve Fail-Closed Durumu", expanded=True):
    st.write(f"**Durum:** `{health}`")
    st.write(f"**Allocation kaynağı:** `{source}`")
    if str(latest.get("previous_valid_date", "")):
        st.write(f"**Önceki geçerli kayıt:** `{latest.get('previous_valid_date')}`")
    issues = str(latest.get("data_health_issues", ""))
    warnings = str(latest.get("data_health_warnings", ""))
    if issues:
        st.write("**Issues:**", issues)
    if warnings:
        st.write("**Warnings:**", warnings)

with st.expander("📈 Kısa Tarihsel İz"):
    hist_cols = [c for c in ["date", "regime_name", "decision_status", "data_health_status", "oil_event_score", "commodity_weight", "allocation_total"] if c in df.columns]
    st.dataframe(df[hist_cols].tail(30), use_container_width=True, hide_index=True)

with st.expander("🧪 Saklanan Backtest / Stres Sonuçları"):
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            st.json(cfg.get("backtest_metrics", {}))
        except Exception as exc:
            st.warning(f"Config okunamadı: {exc}")
    if os.path.exists(STRESS_FILE):
        try:
            with open(STRESS_FILE, "r", encoding="utf-8") as f:
                stress = json.load(f)
            st.json({
                "sensitivity_top_calmar": stress.get("sensitivity_top_calmar", [])[:10],
                "sensitivity_top_sharpe": stress.get("sensitivity_top_sharpe", [])[:10],
            })
        except Exception as exc:
            st.warning(f"Stres sonucu okunamadı: {exc}")

st.caption(
    "Not: Backtest/stres sonuçları tarihsel test çıktılarıdır; canlı veri doğrulaması değildir. "
    "Canlı allocation yalnızca sağlıklı ve yeterli veri bulunduğunda değiştirilir."
)
