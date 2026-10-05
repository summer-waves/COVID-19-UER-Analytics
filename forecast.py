"""Small, honest unemployment forecaster: Ridge regression on lagged features."""
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import CONTEXT_COLS


def _model():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))


def _covs(g: pd.DataFrame) -> list[str]:
    return [c for c in CONTEXT_COLS if c in g and g[c].notna().all()]


def _design(g: pd.DataFrame, covs: list[str]):
    """Each row predicts this month's unemployment from LAST month(s) only."""
    u = g["unemployment_rate"]
    logc = np.log1p(g["new_covid_cases"])
    X = pd.DataFrame({
        "u_lag1": u.shift(1), "u_lag2": u.shift(2),
        "logc_lag1": logc.shift(1), "logc_lag2": logc.shift(2),
    })
    for c in covs:
        X[f"{c}_lag1"] = g[c].shift(1)
    ok = X.notna().all(axis=1)
    return X[ok], (u - X["u_lag1"])[ok], g.loc[ok, "date"]


def backtest(g: pd.DataFrame, n_test: int = 6):
    """Walk-forward: refit on everything before month i, predict month i. Compare to 'same as last month'."""
    g = g.sort_values("date").reset_index(drop=True)
    X, y, dates = _design(g, _covs(g))
    n_test = min(n_test, len(X) - 8)
    if n_test < 1:
        return None
    rows = []
    for i in range(len(X) - n_test, len(X)):
        m = _model().fit(X.iloc[:i], y.iloc[:i])
        last = float(X.iloc[i]["u_lag1"])
        rows.append({"date": dates.iloc[i], "actual": last + float(y.iloc[i]),
                     "model": last + float(m.predict(X.iloc[[i]])[0]),
                     "naive": last})
    out = pd.DataFrame(rows)
    out["model_err"] = (out["model"] - out["actual"]).abs()
    out["naive_err"] = (out["naive"] - out["actual"]).abs()
    return out


def forecast(g: pd.DataFrame, horizon: int = 3, case_multiplier: float = 1.0):
    """Recursive multi-step forecast. Future cases are a SCENARIO: last-3-month average x multiplier."""
    g = g.sort_values("date").reset_index(drop=True)
    covs = _covs(g)
    X, y, _ = _design(g, covs)
    model = _model().fit(X, y)

    bt = backtest(g)
    rmse = float(np.sqrt(((bt["model"] - bt["actual"]) ** 2).mean())) if bt is not None else float(y.std())

    u = list(g["unemployment_rate"])
    logc = list(np.log1p(g["new_covid_cases"]))
    scenario_cases = max(g["new_covid_cases"].tail(3).mean() * case_multiplier, 0)
    last_cov = {c: g[c].iloc[-1] for c in covs}   # context variables held at their last value

    preds = []
    for _ in range(horizon):
        row = {"u_lag1": u[-1], "u_lag2": u[-2], "logc_lag1": logc[-1], "logc_lag2": logc[-2]}
        row.update({f"{c}_lag1": v for c, v in last_cov.items()})
        p = max(u[-1] + float(model.predict(pd.DataFrame([row])[X.columns])[0]), 0.0)
        preds.append(p)
        u.append(p)
        logc.append(float(np.log1p(scenario_cases)))

    dates = pd.date_range(g["date"].iloc[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    out = pd.DataFrame({"date": dates, "forecast": preds})
    steps = np.arange(1, horizon + 1)
    out["low"] = (out["forecast"] - 1.28 * rmse * np.sqrt(steps)).clip(lower=0)  # rough ~80% band
    out["high"] = out["forecast"] + 1.28 * rmse * np.sqrt(steps)
    return out, bt
