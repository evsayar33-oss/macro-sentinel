import json
import os

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Macro Sentinel v2.6",
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
    .main { background:#07090e; color:#e1e7ec; }
    .block-container { padding-top:1.0rem; padding-bottom:2.0rem; max-width:1250px; }
    .topbar { padding:10px 14px; border:1px solid #273244; border-radius:10px; background:#0d121b; margin-bottom:14px; }
    .card { padding:18px; border-radius:14px; border:1px solid #273244; background:#0d121b; margin-bottom:14px; }
    .label { font-size:12px; color:#8fa0b5; text-transform:uppercase; letter-spacing:.06em; }
    .value { font-size:24px; font-weight:700; margin-top:4px; }
    .muted { color:#97a6b8; font-size:13px; }
    .ok { color:#3ddc97; }
    .warn { color:#ffd166; }
    .bad { color:#ff6b6b; }
    .accent { color:#82b1ff; }
    .allocation-row { display:flex; flex-wrap:wrap; gap:10px; }
    .allocation-item { flex:1 1 150px; min-width:135px; padding:14px; border:1px solid #273244; border-radius:10px; background:#101722; }
    .allocation-name { color:#a8b4c4; font-size:13px; }
    .allocation-value { font-size:25px; font-weight:800; margin-top:4px; }
    .mini-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
    .mini { padding:12px; border:1px solid #273244; border-radius:10px; background:#101722; }
    @media (max-width: 700px) {
        .block-container { padding-left:.65rem; padding-right:.65rem; }
        .value { font-size:21px; }
        .allocation-item { min-width:120px; flex-basis:120px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fnum(row, key, default=np.nan):
    try:
        x = float(row.get(key, default))
        return x if np.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def bval(row, key, default=False):
    return str(row.get(key, default)).strip().lower() in {"1", "true", "yes", "on"}


def text_or(row, key, default="—"):
    v = row.get(key, default)
    if pd.isna(v) if not isinstance(v, str) else False:
        return default
    s = str(v).strip()
    return s if s and s.lower() != "nan" else default


def pct(v):
    return "—" if not np.isfinite(v) else f"%{v:.2f}"


def zfmt(v):
    return "—" if not np.isfinite(v) else f"{v:.2f}σ"


def ratio_pct(v):
    return "—" if not np.isfinite(v) else f"%{v * 100.0:.0f}"


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

health = text_or(latest, "data_health_status", "UNKNOWN").upper()
decision = text_or(latest, "decision_status", "UNKNOWN").upper()
source = text_or(latest, "allocation_source", "UNKNOWN").upper()
strategy_mode = text_or(latest, "strategy_mode", "STATIC_REGIME")
event_coverage = text_or(latest, "event_coverage_status", "UNKNOWN").upper()
issues = text_or(latest, "data_health_issues", "")
warnings = text_or(latest, "data_health_warnings", "")
regime_id = int(fnum(latest, "regime_id", 0))
regime_name = text_or(latest, "regime_name", "REJIMSIZ_GECIS")
regime_type = text_or(latest, "regime_type", "TRANSITION")
regime_subtype = text_or(latest, "regime_subtype", "Dengeli / Nötr Piyasa")
regime_status = text_or(latest, "regime_status", "")

weights = {
    "Hisse": fnum(latest, "eq_weight", 0.0),
    "Tahvil": fnum(latest, "bond_weight", 0.0),
    "Nakit": fnum(latest, "cash_weight", 0.0),
    "Altın": fnum(latest, "gold_weight", 0.0),
    "Emtia": fnum(latest, "commodity_weight", 0.0),
    "Kripto": fnum(latest, "crypto_weight", 0.0),
}
allocation_total = sum(v for v in weights.values() if np.isfinite(v))

oil_score = fnum(latest, "oil_event_score", 0.0)
oil_pressure = fnum(latest, "oil_pressure_score", max(fnum(latest, "oil_momentum_score", 0.0), fnum(latest, "oil_structural_score", 0.0)))
oil_active = bval(latest, "oil_event_active")
oil_overlay_applied = bval(latest, "commodity_event_active")
oil_structural_qualified = bval(latest, "oil_structural_qualified")
oil_momentum_confirmed = bval(latest, "oil_momentum_confirmed")
oil_momentum = fnum(latest, "oil_momentum_score", 0.0)
oil_structural = fnum(latest, "oil_structural_score", 0.0)
oil_event_type = text_or(latest, "oil_event_type", "NONE")
unknown_score = fnum(latest, "unknown_event_score", 0.0)
unknown_active = bval(latest, "unknown_event_active")
vix3m = fnum(latest, "vix_term", np.nan)

# Health semantics: optional missing data should not visually imply that the
# current decision is invalid when no critical issue blocked the engine.
st.caption(f"Strateji profili: **{strategy_mode}** · Olay kapsamı: **{event_coverage}**")

if health == "HALT":
    health_label = "🔴 HALT"
    decision_health = "BLOCKED"
    health_color = "bad"
elif health == "MARKET_CLOSED":
    health_label = "🔵 MARKET CLOSED · HOLD"
    decision_health = "HOLD"
    health_color = "accent"
elif issues:
    health_label = "🟠 DEGRADED · KRİTİK VERİ"
    decision_health = "CAUTION" if decision == "LIVE" else decision
    health_color = "warn"
else:
    health_label = "🟢 HEALTHY" if health == "HEALTHY" else "🟡 " + health
    decision_health = "VALID" if decision == "LIVE" else decision
    health_color = "ok" if health == "HEALTHY" else "warn"

if regime_type == "SHOCK":
    hero_color = "#ff5252"
elif regime_type == "RISK_ON":
    hero_color = "#00e676"
elif regime_type == "CONSTRAINT":
    hero_color = "#ffd166"
else:
    hero_color = "#ffd600"

st.title("🏛️ Macro Sentinel")
st.caption("Point-in-time makro rejim + bağımsız olay sensörleri + fail-closed allocation")

st.markdown(
    f"""
    <div class="topbar">
      <b>Veri:</b> <span class="{health_color}">{health_label}</span>
      &nbsp; | &nbsp; <b>Karar:</b> <span class="accent">{decision}</span>
      &nbsp; | &nbsp; <b>Karar Sağlığı:</b> <span class="{('ok' if decision_health == 'VALID' else 'warn' if decision_health == 'CAUTION' else 'bad')}">{decision_health}</span>
      &nbsp; | &nbsp; <b>Kaynak:</b> <span class="accent">{source}</span>
      &nbsp; | &nbsp; <b>Son Güncelleme:</b> <span class="accent">{text_or(latest, 'date', 'N/A')}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if health == "HALT":
    st.error(
        "Yeni allocation engellendi. Sistem son geçerli allocation'ı koruyor; ilk geçerli kayıt yoksa %100 nakit güvenlik durumunda."
    )
    if issues:
        st.write(f"**Kritik veri sorunları:** {issues}")
elif warnings:
    st.warning(f"İkincil veri notu: {warnings}")

# Explicit distinction between regime and event.
st.markdown(
    f"""
    <div class="card" style="border:2px solid {hero_color};">
      <div class="label">CURRENT CONFIRMED REGIME</div>
      <div class="value" style="color:{hero_color};">{regime_name}</div>
      <div class="muted" style="margin-top:6px;">ID {regime_id} · {regime_type} · {regime_subtype}</div>
      <div class="muted" style="margin-top:8px;"><b>State:</b> {regime_status}</div>
      <div class="muted" style="margin-top:4px;"><b>Regime ve event ayrı katmanlardır:</b> mevcut rejim başka, bağımsız olağanüstü olay sensörleri başka ölçülür.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("🎯 Ana Sensörler")
metric_cols = st.columns(5)
metric_values = [
    ("CMS", f"{fnum(latest, 'cms', 0.0):.2f}"),
    ("Petrol Event", "AKTİF" if oil_active else "PASİF"),
    ("Petrol 20g", zfmt(fnum(latest, "oil_ret_20d_z"))),
    ("VIX Z", zfmt(fnum(latest, "vix_z"))),
    ("NDL Z", zfmt(fnum(latest, "ndl_z"))),
]
for col, (label, value) in zip(metric_cols, metric_values):
    with col:
        st.metric(label, value)

st.subheader("🛢️ Petrol / Yapısal Olay Katmanı")
oil_col, structural_col, unknown_col, breadth_col, quality_col = st.columns(5)
with oil_col:
    st.metric("Petrol Event", "AKTİF" if oil_active else "PASİF", delta=f"Event {oil_score:.2f} · Baskı {oil_pressure:.2f}")
with structural_col:
    display_type = {"PRESSURE_ONLY": "YAPISAL BASKI", "NONE": "YOK", "STRUCTURAL": "YAPISAL EVENT", "MOMENTUM": "MOMENTUM EVENT", "COMBINED": "BİRLEŞİK EVENT"}.get(oil_event_type, oil_event_type)
    st.metric("Olay Durumu", display_type, help="YAPISAL BASKI = pressure var fakat teyit yok. YAPISAL EVENT = yapısal teyit var. MOMENTUM EVENT = kısa vadeli teyit var. BİRLEŞİK EVENT = ikisi de teyitli.")
with unknown_col:
    st.metric("Unknown Anomaly", "AKTİF" if unknown_active else "PASİF", delta=f"Skor {unknown_score:.2f}")
with breadth_col:
    breadth = fnum(latest, "commodity_breadth_20d", 0.0)
    energy_breadth = fnum(latest, "energy_breadth_20d", np.nan)
    st.metric("Emtia Genişliği 20g", ratio_pct(breadth), help="Bakır/gümüş/altın 20 günlük pozitif katılım.")
    st.caption(f"Enerji genişliği: {ratio_pct(energy_breadth)}")
with quality_col:
    quality = fnum(latest, "oil_event_quality_60d", 0.5)
    st.metric("Petrol Event Kalitesi", ratio_pct(quality), help="Geçmiş petrol olaylarının 5 günlük pozitif takip davranışının basit tarihsel ölçümü.")

struct_c1, struct_c2, struct_c3, struct_c4 = st.columns(4)
with struct_c1:
    st.metric("Momentum Skoru", f"{oil_momentum:.2f}")
with struct_c2:
    st.metric("Yapısal Skor", f"{oil_structural:.2f}")
with struct_c3:
    level_pct = fnum(latest, "oil_level_percentile_756", np.nan)
    st.metric("WTI 3Y Fiyat Persentili", "—" if not np.isfinite(level_pct) else f"%{level_pct:.1f}")
with struct_c4:
    persist = fnum(latest, "oil_high_level_persistence_60d", np.nan)
    st.metric("Yüksek Seviye Sürekliliği", ratio_pct(persist))

if oil_active:
    st.success(
        f"🛢️ Doğrulanmış petrol olayı aktif — {oil_event_type}. "
        f"Momentum {oil_momentum:.2f} / Yapısal {oil_structural:.2f}. "
        f"Allocation overlay: {'AKTİF' if oil_overlay_applied else 'DEVREDE DEĞİL'}. {text_or(latest, 'commodity_event_reason', '')}"
    )
else:
    st.caption(
        f"Petrol event pasif. Baskı={oil_pressure:.2f} · Momentum={oil_momentum:.2f} · Yapısal={oil_structural:.2f}. "
        f"Yapısal qualification={'EVET' if oil_structural_qualified else 'HAYIR'}. "
        "Pressure-only durumları allocation değiştirmez; yalnızca izlenir."
    )
if unknown_active:
    st.warning(
        f"⚠️ Tanımlanmamış anomali sensörü aktif (skor {unknown_score:.2f}). "
        "Sistem yön tahmini uydurmak yerine risk azaltıcı guard uygulayabilir."
    )

hc1, hc2, hc3 = st.columns(3)
with hc1:
    st.metric("Core Veri", "SAĞLAM" if health == "HEALTHY" else health)
with hc2:
    st.metric("Event Kapsamı", event_coverage)
with hc3:
    st.metric("Strateji", strategy_mode)

st.subheader("⚖️ Mevcut Allocation")
alloc_html = '<div class="allocation-row">'
for name, value in weights.items():
    alloc_html += (
        f'<div class="allocation-item"><div class="allocation-name">{name}</div>'
        f'<div class="allocation-value">{pct(value)}</div></div>'
    )
alloc_html += "</div>"
st.markdown(alloc_html, unsafe_allow_html=True)

if abs(allocation_total - 100.0) > 0.05:
    st.error(f"Allocation bütünlüğü bozuk: {pct(allocation_total)}")
else:
    st.success(f"✅ Allocation bütünlüğü: %{allocation_total:.2f}")

st.caption(
    f"Allocation kaynağı: {source}. "
    + (f"Son geçerli kayıt: {text_or(latest, 'previous_valid_date', 'Yok/ilk geçerli kayıt yok')}." if source != "LIVE_ENGINE" else "Karar mevcut point-in-time canlı motor çıktısından kaydedildi.")
)

st.subheader("🧠 Bu Rejim Neden Aktif?")
regime_reason = []
regime_reason.append(f"Rejim ID: {regime_id} — {regime_name}")
regime_reason.append(f"Alt tip: {regime_subtype}")
severity_source = text_or(latest, "regime_severity_source", "NONE")
regime_reason.append(f"Rejim şiddet kaynağı: {severity_source}")
pending_id = int(fnum(latest, "pending_regime_id", 0))
pending_count = int(fnum(latest, "pending_regime_count", 0))
if pending_id:
    regime_reason.append(f"Bekleyen rejim teyidi: R{pending_id} · {pending_count} gözlem / gerekli teyit adımı")
conflict_note = text_or(latest, "conflict_note", "Açıklama kaydı yok.")
regime_reason.append(f"Karar motoru açıklaması: {conflict_note}")
for line in regime_reason:
    st.write(f"• {line}")

st.subheader("📡 Sensör Matrisi")
sensor_rows = [
    ["Petrol 5g Getiri", zfmt(fnum(latest, "oil_ret_5d_z")), "Olay katmanı"],
    ["Petrol 20g Getiri", zfmt(fnum(latest, "oil_ret_20d_z")), "Olay katmanı"],
    ["Petrol 5g Anomali Persentili", f"{fnum(latest, 'oil_abs_5d_percentile', 50.0):.1f}", "Uyarlanabilir eşik"],
    ["Petrol Fiyat Persentili 252g", f"{fnum(latest, 'oil_level_percentile_252', 0.0):.1f}", "Kısa/orta vadeli seviye"],
    ["Petrol Fiyat Persentili 756g", f"{fnum(latest, 'oil_level_percentile_756', 0.0):.1f}", "Yapısal seviye (~3 yıl)"],
    ["Petrol Yüksek-Seviye Sürekliliği", ratio_pct(fnum(latest, 'oil_high_level_persistence_60d', 0.0)), "Yapısal süreklilik"],
    ["Brent Fiyat Persentili 252g", f"{fnum(latest, 'brent_level_percentile_252', 0.0):.1f}", "İkinci petrol benchmarkı"],
    ["Brent-WTI Spread", f"{fnum(latest, 'brent_wti_spread', 0.0):.2f}", "Bölgesel dislokasyon"],
    ["Brent-WTI Spread Persentili", f"{fnum(latest, 'brent_wti_spread_percentile_252', 0.0):.1f}", "Dislokasyon anomalisi"],
    ["Enerji Genişliği 20g", ratio_pct(fnum(latest, 'energy_breadth_20d', 0.0)), "Enerji kompleks teyidi"],
    ["Crude Inventory Draw Z", zfmt(fnum(latest, 'crude_inventory_draw_z', np.nan)), "EIA arz baskısı (opsiyonel)"],
    ["Petrol Baskı Skoru", f"{oil_pressure:.2f}", "Confirmed gate öncesi baskı"],
    ["Petrol Momentum Skoru", f"{oil_momentum:.2f}", "Kısa vadeli şok"],
    ["Petrol Yapısal Skoru", f"{oil_structural:.2f}", "Kalıcı yüksek fiyat/arz stresi"],
    ["Petrol Yapısal Teyit", "EVET" if oil_structural_qualified else "HAYIR", "Confirmed event gate"],
    ["Petrol Olay Tipi", "YAPISAL BASKI" if oil_event_type == "PRESSURE_ONLY" else oil_event_type, "Momentum / Structural / Combined"],
    ["Petrol Event Durumu", ("AKTİF" if oil_active else "PASİF") + f" · skor {oil_score:.2f}", "Olay katmanı"],
    ["Petrol Event Kalitesi", ratio_pct(fnum(latest, "oil_event_quality_60d", 0.5)), "Tarihsel event başarısı"],
    ["Emtia Genişliği 20g", ratio_pct(fnum(latest, "commodity_breadth_20d", 0.0)), "Teyit"],
    ["Unknown Anomaly", ("AKTİF" if unknown_active else "PASİF") + f" · skor {unknown_score:.2f}", "Anomali guard"],
    ["Navlun", zfmt(fnum(latest, "freight_lvl_z")), "Stagflasyon sensörü"],
    ["HY OAS", zfmt(fnum(latest, "hy_oas_z")), "Kredi stresi"],
    ["Hisse/Tahvil Korelasyonu", f"{fnum(latest, 'spx_bond_corr', 0.0):.2f}", "R1 teyit"],
    ["DXY 5g", zfmt(fnum(latest, "broad_dollar_5d_z")), "Likidite/carry"],
    ["USDJPY 1g", zfmt(fnum(latest, "usdjpy_1d_z")), "Carry"],
    ["10Y TIPS 1g", zfmt(fnum(latest, "tips_1d_z")), "Reel faiz günlük şok"],
    ["10Y TIPS Seviye Z", zfmt(fnum(latest, "tips_level_z")), "Reel faiz kalıcı baskı"],
    ["10Y TIPS Persentili", f"{fnum(latest, 'tips_level_percentile_252', 50.0):.1f}", "Reel faiz seviye dağılımı"],
    ["T10YIE", zfmt(fnum(latest, "t10yie_z")), "Enflasyon beklentisi"],
    ["NDL", zfmt(fnum(latest, "ndl_z")), "Likidite"],
    ["VIX", zfmt(fnum(latest, "vix_z")), "Volatilite"],
    ["VIX Term", "N/A" if not np.isfinite(vix3m) else f"{vix3m:.3f}", "İkincil sensör"],
]
st.dataframe(pd.DataFrame(sensor_rows, columns=["Sensör", "Değer", "Rol"]), use_container_width=True, hide_index=True)

st.subheader("🔗 Veri Kaynakları / Fallback Zinciri")
source_rows = [
    ["WTI / petrol", text_or(latest, "oil_price_source", "Bilinmiyor")],
    ["Brent", text_or(latest, "brent_price_source", "Bilinmiyor")],
    ["VIX 3M", text_or(latest, "vix3m_source", "Bilinmiyor")],
    ["EIA crude inventory", text_or(latest, "eia_inventory_source", "Kullanılmıyor / key yok")],
]
st.dataframe(pd.DataFrame(source_rows, columns=["Sensör", "Aktif kaynak"]), use_container_width=True, hide_index=True)


st.subheader("🔎 Günlük Model Kalitesi Denetimi")
audit_rows = [
    ["Audit durumu", text_or(latest, "daily_audit_status", "INSUFFICIENT_HISTORY")],
    ["Tamamlanan petrol event'i", text_or(latest, "audit_completed_oil_events_5d", "0")],
    ["Petrol event 5g pozitif takip", "—" if not np.isfinite(fnum(latest, "audit_oil_positive_rate_5d", np.nan)) else ratio_pct(fnum(latest, "audit_oil_positive_rate_5d"))],
    ["Tamamlanan yapısal event", text_or(latest, "audit_completed_structural_events_5d", "0")],
    ["Yapısal event 5g pozitif takip", "—" if not np.isfinite(fnum(latest, "audit_structural_positive_rate_5d", np.nan)) else ratio_pct(fnum(latest, "audit_structural_positive_rate_5d"))],
    ["Tamamlanan momentum event", text_or(latest, "audit_completed_momentum_events_5d", "0")],
    ["Momentum event 5g pozitif takip", "—" if not np.isfinite(fnum(latest, "audit_momentum_positive_rate_5d", np.nan)) else ratio_pct(fnum(latest, "audit_momentum_positive_rate_5d"))],
]
st.dataframe(pd.DataFrame(audit_rows, columns=["Kontrol", "Değer"]), use_container_width=True, hide_index=True)
st.caption("Bu audit yalnızca gerçekleşmiş geçmiş event'lerin sonraki 5 işlem günündeki sonuçlarını ölçer; bugünkü kararı geriye dönük olarak değiştirmez.")

st.subheader("🛡️ Veri Sağlığı ve Fail-Closed")
health_rows = [
    ["Core veri sağlığı", health],
    ["Event veri kapsamı", text_or(latest, "event_coverage_status", "UNKNOWN")],
    ["İkincil veri", "1+ uyarı" if warnings else "Tam"],
    ["Karar durumu", decision],
    ["Karar sağlığı", decision_health],
    ["Allocation kaynağı", source],
    ["Son geçerli kayıt", text_or(latest, "previous_valid_date", "Yok / ilk geçerli kayıt yok")],
    ["Kritik sorun", issues or "Yok"],
    ["İkincil uyarı", warnings or "Yok"],
]
st.dataframe(pd.DataFrame(health_rows, columns=["Kontrol", "Durum"]), use_container_width=True, hide_index=True)

with st.expander("📈 Kısa Tarihsel İz", expanded=False):
    hist_cols = [
        c for c in [
            "date", "regime_name", "decision_status", "data_health_status",
            "oil_event_score", "commodity_weight", "eq_weight", "cash_weight", "allocation_total"
        ] if c in df.columns
    ]
    if hist_cols:
        st.dataframe(df[hist_cols].tail(30), use_container_width=True, hide_index=True)

with st.expander("🧪 Saklanan Backtest / Stres Sonuçları", expanded=False):
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
    "Backtest/stres sonuçları tarihsel test çıktılarıdır; canlı doğrulama değildir. "
    "Canlı allocation yalnızca yeterli kritik veri mevcut olduğunda değiştirilir."
)
