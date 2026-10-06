import sys
import numpy as np
import pandas as pd

from forecast import _design, _covs, _model


def run(g, drop_start="2020-04-01", drop_end="2020-07-01", n_test=6):
    g = g.sort_values("date").reset_index(drop=True)
    X, y, dates = _design(g, _covs(g))
    dates = dates.reset_index(drop=True); X = X.reset_index(drop=True); y = y.reset_index(drop=True)
    keep = ~dates.between(pd.Timestamp(drop_start), pd.Timestamp(drop_end))
    rows = []
    for i in range(len(X) - n_test, len(X)):
        train = np.arange(i)                       # all rows before month i
        train_x = train[keep.iloc[train].values]   # same, minus the spike months
        last = float(X.iloc[i]["u_lag1"])
        full = _model().fit(X.iloc[train], y.iloc[train]).predict(X.iloc[[i]])[0]
        trim = _model().fit(X.iloc[train_x], y.iloc[train_x]).predict(X.iloc[[i]])[0]
        rows.append({"date": dates.iloc[i], "actual": last + y.iloc[i],
                     "model_all": last + full, "model_trimmed": last + trim, "naive": last})
    out = pd.DataFrame(rows)
    for c in ("model_all", "model_trimmed", "naive"):
        out[c + "_err"] = (out[c] - out["actual"]).abs()
    return out


if __name__ == "__main__":
    from data_prep import load_all
    country = sys.argv[1] if len(sys.argv) > 1 else "India"
    df = load_all()
    res = run(df[df["country"] == country])
    print(f"\n{country}: mean absolute error over the last {len(res)} months (percentage points)")
    print(res[["model_all_err", "model_trimmed_err", "naive_err"]].mean().round(2).to_string())
    print("\nMonth by month:")
    print(res.assign(date=res["date"].dt.strftime("%b %Y")).round(2).to_string(index=False))
