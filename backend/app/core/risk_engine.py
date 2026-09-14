"""
echoX - Risk Engine
Decouples raw model logits from security policies and provides:
1. Normalized 0-100 Impersonation Risk Score.
2. 3-tier Decision Policy (Low, Medium, High Risk).
3. Frame-by-frame risk assessment.
"""

from typing import Dict, Any
from backend.app.core.config import RISK_CONFIG


class RiskVerdict:
    LOW_RISK = "LOW_RISK"
    MEDIUM_RISK = "MEDIUM_RISK"
    HIGH_RISK = "HIGH_RISK"


class SecurityAction:
    ALLOW = "ALLOW"
    CHALLENGE_MFA = "CHALLENGE_MFA"
    BLOCK_ALERT = "BLOCK_ALERT"


class RiskEngine:
    """
    Maps soft probability to 0-100 risk score and security policies.
    """

    def compute_risk_score(self, spoof_probability: float) -> float:
        """
        Converts soft spoof probability (0.0 to 1.0) to risk score (0.0 to 100.0).
        """
        clamped = max(0.0, min(1.0, float(spoof_probability)))
        return round(clamped * 100.0, 2)

    def evaluate_decision(self, risk_score: float) -> Dict[str, Any]:
        """
        Evaluates risk score against the 3-tier decision matrix:
        - 0-30: Low Risk -> ALLOW (Bona fide human)
        - 31-60: Medium Risk -> CHALLENGE_MFA (Indeterminate artifacts)
        - 61-100: High Risk -> BLOCK_ALERT (Spoofed / cloned voice)
        """
        if risk_score <= RISK_CONFIG.LOW_MAX:
            verdict = RiskVerdict.LOW_RISK
            action = SecurityAction.ALLOW
            description = "Bona fide human voice verified. Access allowed."
            color = "#10B981"
        elif risk_score <= RISK_CONFIG.MED_MAX:
            verdict = RiskVerdict.MEDIUM_RISK
            action = SecurityAction.CHALLENGE_MFA
            description = "Acoustic ambiguity detected. Triggering secondary MFA challenge."
            color = "#F59E0B"
        else:
            verdict = RiskVerdict.HIGH_RISK
            action = SecurityAction.BLOCK_ALERT
            description = "Synthetic speech or cloned voice detected. Access blocked."
            color = "#EF4444"

        return {
            "risk_score": risk_score,
            "verdict": verdict,
            "action": action,
            "color": color,
            "description": description
        }


RISK_ENGINE = RiskEngine()
