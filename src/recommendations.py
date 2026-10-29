"""Rule-based, explainable recommendations and risk banding."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

LOW_THRESHOLD = 0.25   # risk probability < 25%  -> low risk
HIGH_THRESHOLD = 0.50  # risk probability >= 50% -> high risk


@dataclass(frozen=True)
class RiskBand:
    label: str
    color: str
    emoji: str


BANDS = {
    "low": RiskBand("Düşük Risk", "#2ecc71", "🟢"),
    "medium": RiskBand("Orta Risk", "#f1c40f", "🟡"),
    "high": RiskBand("Yüksek Risk", "#e74c3c", "🔴"),
}


def classify_risk(risk_probability: float) -> RiskBand:
    if risk_probability < LOW_THRESHOLD:
        return BANDS["low"]
    if risk_probability < HIGH_THRESHOLD:
        return BANDS["medium"]
    return BANDS["high"]


@dataclass(frozen=True)
class Rule:
    name: str
    condition: Callable[[Dict], bool]
    message: Callable[[Dict], str]


def _max_loan_for_lti(p: Dict, lti: float = 3.0) -> float:
    return p["monthly_income"] * 12 * lti


RULES: List[Rule] = [
    Rule(
        "high_dti",
        lambda p: p["debt_to_income"] > 0.40,
        lambda p: (
            f"Borç/gelir oranınız %{p['debt_to_income']*100:.0f}. Bu oranı %35'in altına "
            "çekmek için mevcut borçlarınızı kapatmayı veya yapılandırmayı değerlendirin."
        ),
    ),
    Rule(
        "loan_vs_income",
        lambda p: p["loan_amount"] > _max_loan_for_lti(p),
        lambda p: (
            f"Talep ettiğiniz tutar yıllık gelirinizin {p['loan_amount']/(p['monthly_income']*12):.1f} "
            f"katı. Tutarı yaklaşık {_max_loan_for_lti(p):,.0f} seviyesine düşürmek onay şansını artırır."
        ),
    ),
    Rule(
        "short_employment",
        lambda p: p["employment_years"] < 2,
        lambda p: "Çalışma süreniz 2 yılın altında. Daha uzun iş geçmişi veya kefil eklemek riski azaltır.",
    ),
    Rule(
        "no_savings",
        lambda p: p["savings_level"] in ("none", "low"),
        lambda p: "Birikim seviyeniz düşük. Düzenli birikim yapmak veya peşinat artırmak olumlu etki eder.",
    ),
    Rule(
        "long_term",
        lambda p: p["loan_term_months"] > 48 and p["debt_to_income"] > 0.30,
        lambda p: "Uzun vade ve yüksek borçluluk birlikte riski artırıyor. Vadeyi kısaltmayı deneyin.",
    ),
    Rule(
        "renting",
        lambda p: p["housing"] != "own" and p["loan_amount"] > p["monthly_income"] * 24,
        lambda p: "Konut sahibi olmamanız yüksek tutarlı taleplerde risk faktörüdür; ek teminat sunmayı düşünün.",
    ),
]


def generate_recommendations(profile: Dict) -> List[str]:
    """Return a list of actionable advice strings for the given applicant profile."""
    advice = [r.message(profile) for r in RULES if r.condition(profile)]
    if not advice:
        advice.append("Profiliniz güçlü görünüyor. Mevcut finansal disiplininizi sürdürün.")
    return advice