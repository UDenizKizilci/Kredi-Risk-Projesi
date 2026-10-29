"""Streamlit dashboard for the Credit Risk & Scoring Engine."""
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.preprocessing import HOUSING_OPTIONS, RAW_FEATURES, SAVINGS_OPTIONS
from src.recommendations import classify_risk, generate_recommendations

MODEL_PATH = Path(__file__).parent / "models" / "credit_model.pkl"

st.set_page_config(page_title="Credit Risk Engine", page_icon="🏦", layout="wide")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def gauge(approval_pct: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=approval_pct,
            number={"suffix": "%"},
            title={"text": "Onay İhtimali"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 50], "color": "#fadbd8"},
                    {"range": [50, 75], "color": "#fcf3cf"},
                    {"range": [75, 100], "color": "#d5f5e3"},
                ],
            },
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def sidebar_inputs() -> dict:
    st.sidebar.header("📋 Başvuru Bilgileri")
    return {
        "age": st.sidebar.slider("Yaş", 18, 75, 35),
        "monthly_income": st.sidebar.number_input("Aylık Gelir", 1000, 100000, 8000, step=500),
        "loan_amount": st.sidebar.number_input("Talep Edilen Kredi Tutarı", 1000, 1000000, 60000, step=1000),
        "loan_term_months": st.sidebar.select_slider("Vade (ay)", [12, 24, 36, 48, 60], value=36),
        "debt_to_income": st.sidebar.slider("Borç / Gelir Oranı", 0.0, 0.9, 0.30, 0.01),
        "employment_years": st.sidebar.slider("Çalışma Süresi (yıl)", 0.0, 40.0, 5.0, 0.5),
        "housing": st.sidebar.selectbox(
            "Konut Durumu", HOUSING_OPTIONS,
            format_func={"own": "Ev sahibi", "rent": "Kiracı", "free": "Ücretsiz"}.get,
        ),
        "savings_level": st.sidebar.selectbox(
            "Birikim Seviyesi", SAVINGS_OPTIONS,
            format_func={"none": "Yok", "low": "Düşük", "medium": "Orta", "high": "Yüksek"}.get,
        ),
    }


def main() -> None:
    st.title("🏦 Automated Loan / Credit Risk & Scoring Engine")
    st.caption("XGBoost tabanlı risk tahmini + kural tabanlı açıklanabilir öneriler")

    bundle = load_model()
    if bundle is None:
        st.error("Model bulunamadı. Önce `python models/train_model.py` komutunu çalıştırın.")
        st.stop()

    profile = sidebar_inputs()
    if not st.sidebar.button("Değerlendir", type="primary", use_container_width=True):
        st.info("Sol menüden bilgileri girip **Değerlendir** butonuna basın.")
        st.stop()

    risk_proba = float(bundle["pipeline"].predict_proba(pd.DataFrame([profile])[RAW_FEATURES])[0, 1])
    approval_pct = (1 - risk_proba) * 100
    band = classify_risk(risk_proba)

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(gauge(approval_pct, band.color), use_container_width=True)
    with right:
        st.markdown(
            f"<div style='padding:18px;border-radius:12px;background:{band.color}22;"
            f"border-left:8px solid {band.color}'>"
            f"<h3 style='margin:0'>{band.emoji} {band.label}</h3>"
            f"<p style='margin:6px 0 0'>Temerrüt olasılığı: <b>%{risk_proba*100:.1f}</b></p></div>",
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        m1.metric("Kredi / Yıllık Gelir", f"{profile['loan_amount']/(profile['monthly_income']*12):.2f}x")
        m2.metric("Model AUC", f"{bundle['test_auc']:.3f}")

    st.subheader("💡 Durum İyileştirme Önerileri")
    for tip in generate_recommendations(profile):
        st.markdown(f"- {tip}")

    st.caption("Bu uygulama eğitim/demo amaçlıdır; gerçek kredi kararı yerine geçmez.")


if __name__ == "__main__":
    main()