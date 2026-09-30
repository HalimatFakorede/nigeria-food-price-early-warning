"""
Nigeria Food Price Early Warning System, Streamlit dashboard.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub -> share.streamlit.io -> pick repo -> app.py
"""
from __future__ import annotations
import json, pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

ROOT = pathlib.Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"
TAB = ROOT / "outputs" / "tables"

GREEN, GOLD, CLAY, INK, GREY = "#1f6b45", "#c8892c", "#a8543a", "#0d1b14", "#8a958e"
AMBER, RED = "#e0a63c", "#b3402c"
LEVEL_COLOR = {"Low": GREEN, "Watch": "#7fa88c", "Elevated": AMBER, "Alert": RED}

st.set_page_config(page_title="Nigeria Food Price Early Warning",
                   page_icon="🚨", layout="wide")
st.markdown("""
<style>
  .block-container{padding-top:2.2rem;max-width:1280px}
  h1,h2,h3{letter-spacing:-.02em}
  [data-testid="stMetricValue"]{font-size:1.6rem}
  .card{border:1px solid #e2ded2;border-radius:12px;padding:14px 16px;background:#fff}
  .lvl{display:inline-block;padding:3px 10px;border-radius:999px;color:#fff;
       font-size:.74rem;font-weight:700;letter-spacing:.04em}
  .src{background:#f2efe6;border-left:3px solid #1f6b45;padding:.7rem 1rem;
       border-radius:6px;font-size:.84rem;color:#33443a}
</style>""", unsafe_allow_html=True)


@st.cache_data
def load():
    nat = pd.read_csv(PROC / "national_monthly.csv", parse_dates=["month"])
    state = pd.read_csv(PROC / "state_monthly.csv", parse_dates=["month"])
    cur = pd.read_csv(TAB / "current_status.csv", parse_dates=["month"])
    seas = pd.read_csv(TAB / "seasonality.csv")
    fc = pd.read_csv(TAB / "forecasts.csv", parse_dates=["month"])
    wf = pd.read_csv(TAB / "walkforward_predictions.csv", parse_dates=["month"])
    res = json.loads((ROOT / "outputs" / "model_results.json").read_text())
    return nat, state, cur, seas, fc, wf, res


nat, state, cur, seas, fc, wf, res = load()
bt = res["backtest"]
asof = cur.month.max()

st.title("🚨 Nigeria Food Price Early Warning System")
st.markdown(
    f"Monitoring **{nat.item.nunique()} staple commodities** across "
    f"**{state.state.nunique()} states** using WFP market price data. "
    f"Volatility regimes from GARCH(1,1); shock risk from a walk-forward validated "
    f"logistic model. **Data as of {asof:%B %Y}.**")

# ---------------------------------------------------------------- status row
alerts = cur[cur.risk_level.isin(["Alert", "Elevated"])]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Commodities at Elevated or Alert", f"{len(alerts)} of {len(cur)}")
c2.metric("Highest risk", cur.iloc[0]["item"], f"{cur.iloc[0]['prob']*100:.0f}% shock probability")
c3.metric("Signal lift (top quintile)",
          f"{bt['learned_walkforward']['lift_top_quintile']:.2f}×",
          help="How much more likely a top-quintile month is to precede a >10% real price rise")
c4.metric("Out-of-sample AUC", f"{bt['learned_walkforward']['auc']:.3f}",
          delta=f"vs {bt['naive_equal_weight']['auc']:.3f} naive")

st.markdown("---")
t1, t2, t3, t4, t5, t6 = st.tabs(
    ["🚦 Current risk", "📈 Prices & volatility", "🗓️ Seasonality", "🗺️ By state",
     "🔬 Does it work?", "📋 Method"])

# ================================================================ TAB 1
with t1:
    st.subheader(f"Risk status, {asof:%B %Y}")
    st.caption("Probability that the real (inflation-adjusted) price rises more than "
               "10% over the next 3 months, from a model trained only on data "
               "available before each prediction date.")
    cols = st.columns(3)
    for i, r in enumerate(cur.itertuples()):
        with cols[i % 3]:
            st.markdown(f"""
<div class="card" style="margin-bottom:12px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <b style="font-size:1.03rem">{r.item}</b>
    <span class="lvl" style="background:{LEVEL_COLOR[r.risk_level]}">{r.risk_level.upper()}</span>
  </div>
  <div style="font-size:2rem;font-weight:700;color:{LEVEL_COLOR[r.risk_level]};margin:.2rem 0">
    {r.prob*100:.0f}%</div>
  <div style="font-size:.8rem;color:#5d6b62;line-height:1.5">
    3-month momentum <b>{r.mom_3m:+.1f}%</b><br>
    volatility <b>{r.vol_ann:.0f}%</b> annualised, regime <b>{r.regime}</b><br>
    seasonal pressure next 3m <b>{r.seasonal_risk:+.1f}%</b>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### 6-month price outlook (SARIMA, 80% interval)")
    pick = st.selectbox("Commodity", sorted(nat.item.unique()), key="fcpick")
    h = nat[nat.item == pick].tail(48)
    f = fc[fc.item == pick]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.month, y=h.price_real_kg, name="actual (real)",
                             line=dict(color=INK, width=2.4)))
    if len(f):
        fig.add_trace(go.Scatter(x=f.month, y=f.hi80, line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=f.month, y=f.lo80, fill="tonexty",
                                 fillcolor="rgba(31,107,69,.16)", line=dict(width=0),
                                 name="80% interval"))
        fig.add_trace(go.Scatter(x=f.month, y=f.forecast_real_kg, name="forecast",
                                 line=dict(color=GREEN, width=2.6, dash="dash")))
    fig.update_layout(height=380, template="simple_white",
                      yaxis_title="NGN/kg, constant Jan-2020 naira",
                      margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ================================================================ TAB 2
with t2:
    st.subheader("Price history and volatility")
    mode = st.radio("Price basis", ["Real (constant Jan-2020 naira)",
                                    "Nominal naira"], horizontal=True)
    col = "price_real_kg" if mode.startswith("Real") else "price_ngn_kg"
    sel = st.multiselect("Commodities", sorted(nat.item.unique()),
                         default=["Rice (local)", "Millet", "Sorghum", "Beans"])
    d = nat[nat.item.isin(sel)]
    fig = px.line(d, x="month", y=col, color="item", template="simple_white")
    fig.update_layout(height=430, yaxis_title="NGN / kg",
                      margin=dict(l=10, r=10, t=30, b=10))
    if not mode.startswith("Real"):
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### GARCH volatility parameters")
    g = pd.DataFrame(res["garch"]).T
    g.index.name = "commodity"
    st.dataframe(g[["alpha", "beta", "persistence", "nu_df", "mean_ann_vol_pct",
                    "latest_ann_vol_pct", "latest_regime", "near_igarch"]].round(3),
                 use_container_width=True)
    st.caption("**Persistence (α+β)** near 1.0 means volatility shocks decay very slowly, "
               "once a market becomes turbulent it stays turbulent for a long time. Four "
               "series are effectively IGARCH (persistence ≈ 1.00), which means their "
               "unconditional variance is not finite and long-horizon volatility forecasts "
               "from them should not be trusted. Flagged rather than hidden.")

# ================================================================ TAB 3
with t3:
    st.subheader("The lean-season signature")
    st.caption("Percentage deviation from each commodity's annual average price, after "
               "removing the long-run trend. This is the predictable part of the year.")
    piv = seas.pivot(index="item", columns="month_num", values="seasonal_pct")
    piv.columns = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = px.imshow(piv, color_continuous_scale="RdYlGn_r",
                    origin="upper", aspect="auto", text_auto=".0f",
                    zmin=-abs(piv.values).max(), zmax=abs(piv.values).max())
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_colorbar_title="% vs avg")
    st.plotly_chart(fig, use_container_width=True)
    peak = (seas.loc[seas.groupby("item").seasonal_pct.idxmax()]
                .assign(month=lambda x: pd.to_datetime(x.month_num, format="%m").dt.strftime("%B"))
                [["item", "month", "seasonal_pct"]]
                .rename(columns={"seasonal_pct": "peak premium %"}))
    st.markdown("**Peak month for each commodity**")
    st.dataframe(peak.round(1).set_index("item"), use_container_width=True)
    st.info("Most cereals peak in **August**, immediately before the main harvest, the "
            "classic West African lean season, when household stocks are exhausted and "
            "the new crop is not yet in. This is the most operationally useful pattern "
            "in the whole dataset: it is large, it is predictable, and it recurs annually, "
            "so procurement, cash transfers and buffer releases can be scheduled against it.")

# ================================================================ TAB 4
with t4:
    st.subheader("Where prices are highest")
    c1, c2 = st.columns(2)
    item = c1.selectbox("Commodity", sorted(state.item.unique()), key="stitem")
    recent = state[(state.item == item)]
    last_m = recent.month.max()
    win = recent[recent.month >= last_m - pd.DateOffset(months=11)]
    agg = (win.groupby("state")
              .agg(price=("price_ngn_kg", "median"), months=("month", "nunique"))
              .query("months >= 3").sort_values("price", ascending=False))
    c2.metric(f"National median, last 12 months", f"₦{win.price_ngn_kg.median():,.0f}/kg")
    fig = px.bar(agg.reset_index(), x="price", y="state", orientation="h",
                 template="simple_white", color="price",
                 color_continuous_scale=["#e3f0e7", GREEN])
    fig.update_layout(height=430, xaxis_title="NGN / kg (12-month median)",
                      yaxis_title="", margin=dict(l=10, r=10, t=30, b=10),
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"States with at least 3 months of {item} observations in the last year. "
               "Differences reflect transport cost, insecurity, and distance from "
               "producing zones as well as local demand.")

# ================================================================ TAB 5
with t5:
    st.subheader("Does the warning signal actually warn?")
    st.markdown(f"""
This is the question that separates a dashboard from a decision tool, so it is
answered explicitly.

**Event definition:** the real price of a commodity rises more than **10%** over the
following **3 months**. Unconditionally this happens
**{bt['learned_walkforward']['base_rate_pct']:.1f}%** of the time.
    """)
    a, b = st.columns(2)
    with a:
        st.markdown("##### Attempt 1, naive equal-weight composite ❌")
        n = bt["naive_equal_weight"]
        st.metric("ROC-AUC", f"{n['auc']:.3f}", delta="barely above chance",
                  delta_color="inverse")
        st.markdown(f"""
Four sensible-looking indicators, momentum, volatility, trend gap, seasonal risk, averaged with equal weights. It barely beat a coin flip.

**Why it failed:** Nigerian staple prices **mean-revert** over 3 months. Momentum
correlates **−0.19** with the next 3 months' return and the trend gap **−0.09**.
The composite gave both a *positive* weight, so it was pointing the wrong way and
cancelling out seasonal risk (**+0.20**), the one component that worked.
        """)
    with b:
        st.markdown("##### Attempt 2, learned, walk-forward validated ✅")
        l = bt["learned_walkforward"]
        st.metric("ROC-AUC", f"{l['auc']:.3f}",
                  delta=f"+{l['auc']-n['auc']:.3f} vs naive")
        st.markdown(f"""
Weights are **learned** by logistic regression rather than assumed, adding
exchange-rate and market-wide contagion features. Validated **walk-forward**:
at each month the model trains only on observations whose outcome had already
been observed, then predicts forward.

**Top-quintile months precede a price shock
{l['top_quintile_hit_rate_pct']:.0f}% of the time versus a
{l['base_rate_pct']:.1f}% base rate, a {l['lift_top_quintile']:.2f}× lift**
across {l['n']} out-of-sample predictions.
        """)

    q = pd.DataFrame(l["quintiles"])
    qn = pd.DataFrame(n["quintiles"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qn.q, y=qn.hit_rate, name=f"Naive (AUC {n['auc']:.2f})",
                             line=dict(color=GREY, width=3), mode="lines+markers"))
    fig.add_trace(go.Scatter(x=q.q, y=q.hit_rate, name=f"Learned (AUC {l['auc']:.2f})",
                             line=dict(color=GREEN, width=3), mode="lines+markers"))
    fig.add_hline(y=l["base_rate_pct"], line_dash="dash", line_color=RED,
                  annotation_text="base rate")
    fig.update_layout(height=380, template="simple_white",
                      xaxis_title="Risk-score quintile (5 = highest predicted risk)",
                      yaxis_title="% followed by >10% real price rise",
                      margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.14))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### What the model learned")
    co = pd.Series(res["coefficients"]).sort_values()
    fig = go.Figure(go.Bar(x=co.values, y=co.index, orientation="h",
                           marker_color=[RED if v < 0 else GREEN for v in co.values]))
    fig.update_layout(height=430, template="simple_white",
                      xaxis_title="Standardised logistic coefficient (log-odds)",
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.success("**The headline driver is `fx_mom_12m`, 12-month naira depreciation.** "
               "Exchange-rate movement dominates every price-based technical indicator "
               "in the model. For imported rice and vegetable oil that is a direct "
               "pass-through; for domestic cereals it works through imported fertiliser, "
               "diesel and transport. Any Nigerian food-price warning system that ignores "
               "FX is missing its single most important variable.")
    st.warning("**Honest limits.** An AUC of ~0.61 is modest. This signal is useful for "
               "prioritising attention across commodities and for timing seasonal "
               "procurement, it is not a precise price forecast, and it should never be "
               "the only input to a food-security decision.")

# ================================================================ TAB 6
with t6:
    st.subheader("Method, sources and limitations")
    st.markdown(f"""
#### Data
| Source | What | Access |
|---|---|---|
| WFP / HDX, *Nigeria: Food Prices* | {nat.item.nunique()} staples, 68 markets, {state.state.nunique()} states, retail & wholesale, 2002, {asof:%Y} | Open CSV |
| World Bank WDI | Nigeria CPI, used to deflate nominal naira to real prices | Open API |
| Derived | NGN/USD exchange rate, computed as naira price ÷ USD price within the same dataset |, |

#### Pipeline
1. **Unit normalisation.** Prices are quoted in `100 KG`, `2.5 KG`, `400 G`, `L`, `KG`… Every price is converted to **NGN per kg** before anything else. Rows with non-mass units are dropped and counted (0.5% of raw).
2. **Outlier control.** Winsorised at the 1st/99th percentile within commodity × price type. Market data contains 10-100× data-entry errors, and a single one will dominate a volatility model.
3. **Aggregation.** Monthly **median** across markets (not mean, medians resist the remaining outliers). Months with fewer than 3 reporting markets are excluded: the final month of the raw file had a single market reporting and would otherwise have produced a fake national price.
4. **Deflation.** Nominal naira ÷ CPI, rebased to constant January 2020 naira.
5. **Volatility.** GARCH(1,1) with Student-t errors on monthly real log returns. Regimes at the 60th/85th percentiles of fitted conditional volatility.
6. **Seasonality.** Log real price on month dummies plus a linear trend.
7. **Features.** Strictly point-in-time: the GARCH model and the seasonal index are **re-estimated on an expanding window at every month**, so no feature has seen the future.
8. **Validation.** Walk-forward from {bt['walkforward_start'][:7]}: train only on observations whose 3-month outcome window had already closed, then predict the next month.

#### Limitations, stated plainly
- **Modest skill.** AUC ≈ {bt['learned_walkforward']['auc']:.2f}. Real discrimination, but not a precise forecast.
- **Retail series begin in 2014.** WFP wholesale data reaches back to 2002 but was discontinued in 2023; the two are kept as separate panels rather than spliced, because retail and wholesale are different price concepts.
- **CPI is annual, interpolated to monthly.** Nigeria publishes monthly CPI; using it would sharpen the real-price series. The annual World Bank series was used because it is openly and reproducibly accessible without scraping.
- **Market coverage is uneven** across states and over time; northern conflict-affected states have gaps precisely when prices are most volatile, a missing-not-at-random problem that will bias any national average.
- **Not causal.** This is a monitoring and prioritisation tool, not a structural model of food price formation.
    """)
    st.markdown('<div class="src">Built by <b>Halimat H. Fakorede</b>, '
                'Agricultural Data Scientist &amp; Operations Analyst · '
                'github.com/HalimatFakorede</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(f"Data: WFP via HDX, World Bank WDI. Analysis and dashboard by "
           f"Halimat H. Fakorede. Data as of {asof:%B %Y}.")
