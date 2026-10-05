"""COVID-19 & Unemployment Rate Analytics - Plotly Dash dashboard.
Run:  python app.py   then open http://127.0.0.1:8050
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output

from data_prep import load_all, available_context
from forecast import forecast, backtest

df = load_all()
COUNTRIES = sorted(df["country"].unique())
CONTEXT = available_context(df)
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
COLOR = {c: PALETTE[i % len(PALETTE)] for i, c in enumerate(COUNTRIES)}
CONTEXT_LABELS = {
    "stringency_index": "Lockdown stringency (0-100)",
    "people_fully_vaccinated_per_hundred": "Fully vaccinated (per 100 people)",
    "stimulus_pct_gdp": "Stimulus (% of GDP)",
}
LAST_DATE = df["date"].max().strftime("%b %Y")
FIRST_DATE = df["date"].min().strftime("%b %Y")

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "COVID-19 & Unemployment Analytics"
server = app.server  # exposes the Flask server for deployment (e.g. gunicorn app:server)


def pick(countries):
    return df[df["country"].isin(countries or COUNTRIES)]


def country_picker(id_, multi=True, value=None):
    return dcc.Dropdown(COUNTRIES, value if value is not None else (COUNTRIES if multi else COUNTRIES[0]),
                        multi=multi, clearable=False, id=id_)


def controls(*children):
    return html.Div(list(children), style={"display": "flex", "gap": "24px", "flexWrap": "wrap",
                                           "alignItems": "end", "margin": "12px 0"})


def labeled(label, comp, width="280px"):
    return html.Div([html.Label(label, style={"fontWeight": "600"}), comp], style={"minWidth": width})

def backtest_summary():
    rows = []
    for c in COUNTRIES:
        bt = backtest(df[df["country"] == c])
        if bt is None:
            continue
        m, n = bt["model_err"].mean(), bt["naive_err"].mean()
        rows.append({"country": c, "model_mae": round(m, 2), "naive_mae": round(n, 2),
                     "winner": "Model" if m < n else "Naive"})
    return pd.DataFrame(rows)


BACKTEST_SUMMARY = backtest_summary()


# ---------------------------------------------------------------- layout
app.layout = html.Div([
    html.H2("COVID-19 and Unemployment Rate Analytics"),
    html.P(f"{', '.join(COUNTRIES)} | {FIRST_DATE} - {LAST_DATE} | by Marco Ortiz & Tania Ortiz",
           style={"color": "#666"}),
    dcc.Tabs(id="tabs", value="overview", children=[
        dcc.Tab(label="Trends", value="overview"),
        dcc.Tab(label="Correlation", value="corr"),
        dcc.Tab(label="Before / during / after peaks", value="peaks"),
        dcc.Tab(label="Policy context", value="context"),
        dcc.Tab(label="Forecast", value="forecast"),
    ]),
    html.Div(id="tab-body", style={"paddingTop": "8px"}),
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "16px",
          "fontFamily": "system-ui, sans-serif"})


@app.callback(Output("tab-body", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "overview":
        return html.Div([
            controls(labeled("Countries", country_picker("ov-countries")),
                     labeled("Cases axis", dcc.RadioItems(["Linear", "Log"], "Linear", id="ov-scale",
                                                          inline=True), "160px")),
            dcc.Graph(id="ov-graph", style={"height": "680px"}),
        ])
    if tab == "corr":
        return html.Div([
            controls(labeled("Countries", country_picker("co-countries")),
                     labeled("Max lag (months)", dcc.Slider(0, 6, 1, value=4, id="co-lag",
                             marks={i: str(i) for i in range(7)}), "300px")),
            html.H4("Pearson r: cases vs unemployment (lag 0 = same month, matches the original notebook)"),
            dcc.Graph(id="co-heat"),
            html.H4("Scatter with linear fit"),
            dcc.Graph(id="co-scatter"),
            html.P("Lag k pairs cases from k months EARLIER with this month's unemployment rate.",
                   style={"color": "#666"}),
        ])
    if tab == "peaks":
        return html.Div([
            controls(labeled("Countries", country_picker("pk-countries")),
                     labeled("Peak definition", dcc.RadioItems(
                         ["Largest peak overall", "Largest peak in first 12 months"],
                         "Largest peak overall", id="pk-def"), "300px"),
                     labeled("Window (+/- months)", dcc.Slider(3, 12, 1, value=6, id="pk-win",
                             marks={i: str(i) for i in (3, 6, 9, 12)}), "300px")),
            dcc.Graph(id="pk-graph", style={"height": "520px"}),
            html.P("0 = the month with the most new cases. Lines show the unemployment rate change "
                   "(percentage points) relative to that month. Lines are shorter where the peak is near "
                   "the edge of the dataset.", style={"color": "#666"}),
        ])
    if tab == "context":
        if not CONTEXT:
            return html.Div([
                html.H4("No policy data loaded yet"),
                html.P("Run  python fetch_context_data.py  to download vaccination + stringency data "
                       "(OWID), and/or add data/stimulus.csv (country,date,stimulus_pct_gdp). "
                       "Then restart the app."),
            ])
        return html.Div([
            controls(labeled("Country", country_picker("cx-country", multi=False)),
                     labeled("Context variable", dcc.Dropdown(
                         [{"label": CONTEXT_LABELS[c], "value": c} for c in CONTEXT], CONTEXT[0],
                         clearable=False, id="cx-var"), "340px")),
            dcc.Graph(id="cx-graph", style={"height": "480px"}),
            dcc.Graph(id="cx-corr"),
        ])
    # forecast
    return html.Div([
        controls(labeled("Country", country_picker("fc-country", multi=False)),
                 labeled("Months ahead", dcc.Slider(1, 6, 1, value=3, id="fc-h",
                         marks={i: str(i) for i in range(1, 7)}), "300px"),
                 labeled("Future cases scenario (x last 3-month avg)", dcc.Slider(
                     0.25, 3, 0.25, value=1, id="fc-mult",
                     marks={0.25: "0.25x", 1: "1x", 2: "2x", 3: "3x"}), "360px")),
        dcc.Graph(id="fc-graph"),
        html.H4("Backtest: last 6 months, one-step-ahead"),
        html.Div(id="fc-table"),
        html.H4("Backtest summary: all countries (mean absolute error, pp)"),
        dash_table.DataTable(
            BACKTEST_SUMMARY.to_dict("records"),
            [{"name": "Country", "id": "country"},
             {"name": "Model error", "id": "model_mae"},
             {"name": "Naive error", "id": "naive_mae"},
             {"name": "Lower error", "id": "winner"}],
            style_cell={"textAlign": "center"}, style_header={"fontWeight": "bold"},
            style_data_conditional=[{"if": {"filter_query": '{winner} = "Model"'},
                                     "backgroundColor": "#e6f4ea"}]),
        html.P("Small-sample model (about 25 monthly rows per country). Use it to explore relationships, "
               "not as a real prediction. The band is a rough range based on backtest error.",
               style={"color": "#666"}),
    ])


# ---------------------------------------------------------------- Trends
@app.callback(Output("ov-graph", "figure"), Input("ov-countries", "value"), Input("ov-scale", "value"))
def trends(countries, scale):
    d = pick(countries)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("New COVID-19 cases", "Unemployment rate (%)"))
    for c, g in d.groupby("country"):
        fig.add_scatter(x=g["date"], y=g["new_covid_cases"], name=c, mode="lines+markers",
                        line_color=COLOR[c], legendgroup=c, row=1, col=1)
        fig.add_scatter(x=g["date"], y=g["unemployment_rate"], name=c, mode="lines+markers",
                        line_color=COLOR[c], legendgroup=c, showlegend=False, row=2, col=1)
    fig.update_yaxes(type="log" if scale == "Log" else "linear", row=1, col=1)
    fig.update_layout(hovermode="x unified", template="plotly_white", margin=dict(t=40))
    return fig


# ---------------------------------------------------------------- Correlation
def lag_corr(g, k):
    s = g.sort_values("date")
    return s["new_covid_cases"].shift(k).corr(s["unemployment_rate"])


@app.callback(Output("co-heat", "figure"), Output("co-scatter", "figure"),
              Input("co-countries", "value"), Input("co-lag", "value"))
def correlation(countries, max_lag):
    d = pick(countries)
    names = list(d["country"].unique())
    z = [[lag_corr(d[d["country"] == c], k) for k in range(max_lag + 1)] for c in names]
    heat = go.Figure(go.Heatmap(z=z, x=[f"lag {k}" for k in range(max_lag + 1)], y=names,
                                zmin=-1, zmax=1, colorscale="RdBu_r", text=np.round(z, 2),
                                texttemplate="%{text}"))
    heat.update_layout(template="plotly_white", height=60 + 70 * len(names), margin=dict(t=10))

    ncols = min(3, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    sc = make_subplots(rows=nrows, cols=ncols, subplot_titles=names)
    for i, c in enumerate(names):
        g = d[d["country"] == c].dropna(subset=["new_covid_cases", "unemployment_rate"])
        r, cc = divmod(i, ncols)
        sc.add_scatter(x=g["new_covid_cases"], y=g["unemployment_rate"], mode="markers",
                       marker=dict(color=COLOR[c], opacity=0.6), name=c, showlegend=False,
                       text=g["date"].dt.strftime("%b %Y"), row=r + 1, col=cc + 1)
        if len(g) > 2:
            m, b = np.polyfit(g["new_covid_cases"], g["unemployment_rate"], 1)
            xs = np.array([g["new_covid_cases"].min(), g["new_covid_cases"].max()])
            sc.add_scatter(x=xs, y=m * xs + b, mode="lines", line=dict(color="red"),
                           showlegend=False, row=r + 1, col=cc + 1)
    sc.update_layout(template="plotly_white", height=330 * nrows)
    sc.update_xaxes(title_text="New cases", title_font_size=11)
    sc.update_yaxes(title_text="Unemployment (%)", title_font_size=11)
    return heat, sc


# ---------------------------------------------------------------- Peaks
@app.callback(Output("pk-graph", "figure"), Input("pk-countries", "value"),
              Input("pk-def", "value"), Input("pk-win", "value"))
def peaks(countries, definition, win):
    fig = go.Figure()
    for c, g in pick(countries).groupby("country"):
        g = g.sort_values("date").reset_index(drop=True)
        search = g[g["date"] < g["date"].min() + pd.DateOffset(months=12)] \
            if definition.startswith("Largest peak in first") else g
        peak_idx = search["new_covid_cases"].idxmax()
        rel = ((g["date"].dt.year - g.loc[peak_idx, "date"].year) * 12
               + g["date"].dt.month - g.loc[peak_idx, "date"].month)
        sel = g[rel.abs() <= win]
        change = sel["unemployment_rate"] - g.loc[peak_idx, "unemployment_rate"]
        fig.add_scatter(x=rel[sel.index], y=change, mode="lines+markers", name=
                        f"{c} (peak {g.loc[peak_idx, 'date']:%b %Y})", line_color=COLOR[c])
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white", xaxis_title="Months from case peak",
                      yaxis_title="Unemployment change vs peak month (pp)", hovermode="x unified")
    return fig


# ---------------------------------------------------------------- Policy context
if CONTEXT:
    @app.callback(Output("cx-graph", "figure"), Output("cx-corr", "figure"),
                  Input("cx-country", "value"), Input("cx-var", "value"))
    def context(country, var):
        g = df[df["country"] == country].sort_values("date")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_scatter(x=g["date"], y=g["unemployment_rate"], name="Unemployment (%)",
                        line_color="#d62728", secondary_y=False)
        fig.add_scatter(x=g["date"], y=g[var], name=CONTEXT_LABELS[var],
                        line_color="#1f77b4", line_dash="dot", secondary_y=True)
        fig.update_yaxes(title_text="Unemployment (%)", secondary_y=False)
        fig.update_yaxes(title_text=CONTEXT_LABELS[var], secondary_y=True)
        fig.update_layout(template="plotly_white", hovermode="x unified")

        rows = {CONTEXT_LABELS[v]: [df[df["country"] == c]["unemployment_rate"].corr(df[df["country"] == c][v])
                                    for c in COUNTRIES] for v in CONTEXT}
        heat = go.Figure(go.Heatmap(z=list(rows.values()), x=COUNTRIES, y=list(rows.keys()),
                                    zmin=-1, zmax=1, colorscale="RdBu_r",
                                    text=np.round(list(rows.values()), 2), texttemplate="%{text}"))
        heat.update_layout(template="plotly_white", title="Correlation with unemployment rate",
                           height=120 + 60 * len(rows))
        return fig, heat


# ---------------------------------------------------------------- Forecast
@app.callback(Output("fc-graph", "figure"), Output("fc-table", "children"),
              Input("fc-country", "value"), Input("fc-h", "value"), Input("fc-mult", "value"))
def forecast_tab(country, horizon, mult):
    g = df[df["country"] == country].sort_values("date")
    fc, bt = forecast(g, horizon, mult)
    fig = go.Figure()
    fig.add_scatter(x=g["date"], y=g["unemployment_rate"], name="Actual", line_color="#333")
    fig.add_scatter(x=fc["date"], y=fc["high"], mode="lines", line_width=0, showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=fc["date"], y=fc["low"], mode="lines", line_width=0, fill="tonexty",
                    fillcolor="rgba(214,39,40,0.18)", name="Approx. range")
    fig.add_scatter(x=fc["date"], y=fc["forecast"], name="Forecast", mode="lines+markers",
                    line=dict(color="#d62728", dash="dash"))
    if bt is not None:
        fig.add_scatter(x=bt["date"], y=bt["model"], name="Backtest prediction", mode="markers",
                        marker=dict(symbol="x", color="#1f77b4"))
    fig.update_layout(template="plotly_white", yaxis_title="Unemployment rate (%)", hovermode="x unified")

    if bt is None:
        return fig, html.P("Not enough history to backtest.")
    table = bt.assign(date=bt["date"].dt.strftime("%b %Y")).round(2)[
        ["date", "actual", "model", "naive", "model_err", "naive_err"]]
    summary = html.P(f"Mean abs. error: model {bt['model_err'].mean():.2f} pp vs "
                     f"'same as last month' {bt['naive_err'].mean():.2f} pp", style={"fontWeight": "600"})
    return fig, [summary, dash_table.DataTable(
        table.to_dict("records"), [{"name": c, "id": c} for c in table.columns],
        style_cell={"textAlign": "center"}, style_header={"fontWeight": "bold"})]


if __name__ == "__main__":
    app.run(debug=True)
