"""SRE pillar 1 - SLO + Error Budget: our reliability promise and how much is left."""
from __future__ import annotations
import threading

_TARGET = 0.99
_WINDOW = "30d"
_BUDGET_TOTAL = 100.0

_sli = None
_slo = None
_ready = False
_lock = threading.Lock()

def _build():
    global _sli, _slo, _ready
    if _ready:
        return _slo
    with _lock:
        if _ready:
            return _slo
        _ready = True
        try:
            from agent_sre import SLO, ErrorBudget
            from agent_sre.slo.indicators import TaskSuccessRate
            _sli = TaskSuccessRate(target=_TARGET, window=_WINDOW)
            _slo = SLO(name="rag-assistant", indicators=[_sli],
                       error_budget=ErrorBudget(total=_BUDGET_TOTAL))
        except Exception as exc:
            print(f"[sre.slo] init failed: {exc}")
            _sli = None
            _slo = None
    return _slo

def record(good: bool) -> None:
    """Record one task outcome (success/failure) against the SLO."""
    if _build() is None:
        return
    _sli.record(1.0 if good else 0.0)
    _slo.record_event(good)

def status() -> dict:
    """Current SLO status: compliance, error budget, burn rate, alerts."""
    slo = _build()
    if slo is None:
        return {}
    data = slo.to_dict()
    budget = data.get("error_budget", {})
    indicators = data.get("indicators", [{}])
    return {
        "status": data.get("status"),
        "compliance": indicators[0].get("compliance") if indicators else None,
        "budget_remaining_percent": budget.get("remaining_percent"),
        "burn_rate": budget.get("burn_rate"),
        "is_exhausted": budget.get("is_exhausted"),
        "firing_alerts": budget.get("firing_alerts", []),
    }
