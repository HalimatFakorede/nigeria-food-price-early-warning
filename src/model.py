"""
Project 2, Step 2: Volatility regimes, seasonality, and a VALIDATED early-warning model.

THE HONEST VERSION OF THIS PROJECT
  The first design was an equal-weighted composite of four sensible-looking
  indicators (momentum, volatility, trend gap, seasonal risk). It did not work:
  ROC-AUC 0.58 and no lift over the base rate. Diagnosing why turned out to be
  the most valuable part of the project.

  Correlation of each component with the NEXT 3 months' real return:
        momentum (3m)      -0.19     <- mean-reverting, not trend-following
        trend gap          -0.09     <- mean-reverting
        conditional vol    -0.03     <- direction-free by construction
        seasonal risk      +0.20     <- the only correctly-signed component

  Nigerian staple food prices MEAN-REVERT over a 3-month horizon: a price that
  has just spiked is more likely to fall back than to keep climbing (harvest
  arrives, traders release stock, substitution kicks in). The naive score gave
  momentum a POSITIVE weight, so it was actively predicting the wrong direction
  and cancelling the one component that worked.

  The fix is not to hand-tune weights until the backtest looks good, that is
  just overfitting with extra steps. The fix is to LEARN the weights with a
  logistic regression validated WALK-FORWARD: at every month, train only on
  data whose outcome was already observable, then predict the next month.

Both models are reported side by side, because "here is what did not work and
why" is the part that demonstrates actual judgement.

Outputs: outputs/model_results.json, outputs/figures/*.png, outputs/tables/*.csv
Run:  python src/model.py
"""
from __future__ import annotations
import json, pathlib, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from arch import arch_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

GREEN, GOLD, CLAY, INK, GREY = "#1f6b45", "#c8892c", "#a8543a", "#0d1b14", "#8a958e"
AMBER, RED = "#e0a63c", "#b3402c"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": .25, "grid.linestyle": "--",
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlepad": 12,
})

CALM_PCT, HIGH_PCT = 60, 85
HORIZON = 3
EVENT_THRESHOLD = 0.10
MIN_TRAIN_OBS = 30          # months of history before a GARCH/feature row is emitted
WALKFORWARD_START = "2019-06-01"
FEATURES = ["mom_3m", "mom_6m", "mom_12m", "trend_gap", "z12",
            "vol_ann", "vol_chg", "seasonal_risk", "pos_12m", "sin_m", "cos_m",
            "fx_mom_3m", "fx_mom_12m", "fx_vol_12m", "mkt_mom_3m", "rel_strength"]
OUT: dict = {}


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print(f"  figure -> outputs/figures/{name}")


def load() -> pd.DataFrame:
    d = pd.read_csv(PROC / "national_monthly.csv", parse_dates=["month"])
    return d.sort_values(["item", "month"]).reset_index(drop=True)


def series_for(g: pd.DataFrame) -> pd.Series:
    """
    Regular monthly real-price series on a complete DatetimeIndex with an explicit
    'MS' frequency. Internal gaps are linearly interpolated; without a complete
    index statsmodels refuses to run SARIMA ("No supported index is available")
    and the seasonal terms become meaningless.
    """
    s = g.set_index("month")["price_real_kg"].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    return s.reindex(full).interpolate(limit_direction="both").asfreq("MS")


def implied_fx() -> pd.Series:
    """
    Monthly NGN/USD exchange rate DERIVED FROM THE PRICE DATA ITSELF.
    WFP reports every observation in both naira and USD, so the ratio of the two
    is the exchange rate WFP applied. This gives a consistent FX series on exactly
    the same calendar as the prices, with no separate source to reconcile.
    Naira depreciation is the single biggest driver of Nigerian food inflation,
    so leaving it out of a food-price model would be a domain error.
    """
    o = pd.read_csv(PROC / "clean_observations.csv.gz", low_memory=False)
    o["month"] = pd.to_datetime(o["month"])
    o = o[(o.usdprice > 0) & (o.price > 0)]
    fx = (o.assign(fx=o.price / o.usdprice).groupby("month")["fx"].median()
            .sort_index())
    full = pd.date_range(fx.index.min(), fx.index.max(), freq="MS")
    return fx.reindex(full).interpolate(limit_direction="both")


# ============================================================================
# 1. GARCH, descriptive regime layer (full sample, clearly labelled as such)
# ============================================================================
def garch_by_commodity(d: pd.DataFrame):
    frames, params = [], {}
    for item, g in d.groupby("item"):
        s = series_for(g)
        r = np.log(s).diff().dropna() * 100
        if len(r) < 60:
            continue
        am = arch_model(r, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
        res = am.fit(disp="off", show_warning=False)
        ann = res.conditional_volatility * np.sqrt(12)
        lo, hi = np.percentile(ann, [CALM_PCT, HIGH_PCT])
        regime = pd.cut(ann, [-np.inf, lo, hi, np.inf],
                        labels=["Calm", "Elevated", "High"])
        frames.append(pd.DataFrame({
            "item": item, "month": r.index, "ret_pct": r.values,
            "cond_vol_ann_pct": ann.values, "regime": regime.astype(str),
            "price_real_kg": s.reindex(r.index).values,
        }))
        p = res.params
        pers = float(p["alpha[1]"] + p["beta[1]"])
        params[item] = {
            "omega": float(p["omega"]), "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "persistence": pers,
            "nu_df": float(p.get("nu", np.nan)), "aic": float(res.aic),
            "mean_ann_vol_pct": float(ann.mean()),
            "latest_ann_vol_pct": float(ann.iloc[-1]),
            "latest_regime": str(regime.iloc[-1]),
            "calm_threshold": float(lo), "high_threshold": float(hi),
            "n_obs": int(len(r)),
            "near_igarch": bool(pers > 0.99),
        }
        print(f"    {item:<16} persistence={pers:.3f}{' (near-IGARCH)' if pers>0.99 else ''}"
              f"  mean vol={ann.mean():6.1f}%  latest={ann.iloc[-1]:6.1f}% ({regime.iloc[-1]})")
    return pd.concat(frames, ignore_index=True), params


# ============================================================================
# 2. SEASONALITY (descriptive) + expanding-window version (for features)
# ============================================================================
def seasonal_index(s: pd.Series) -> dict:
    """Month-of-year effect on log real price, after removing a linear trend."""
    if len(s) < 36:
        return {m: 0.0 for m in range(1, 13)}
    X = pd.get_dummies(s.index.month.astype("category"),
                       prefix="m", drop_first=True, dtype=float)
    X.index = s.index
    X["t"] = np.arange(len(s))
    X = sm.add_constant(X)
    res = sm.OLS(np.log(s.values), X).fit()
    eff = {1: 0.0}
    for m in range(2, 13):
        eff[m] = float(res.params.get(f"m_{m}", 0.0))
    mu = np.mean(list(eff.values()))
    return {m: (v - mu) * 100 for m, v in eff.items()}


def seasonality_table(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item, g in d.groupby("item"):
        for m, v in seasonal_index(series_for(g)).items():
            rows.append({"item": item, "month_num": m, "seasonal_pct": v})
    return pd.DataFrame(rows)


# ============================================================================
# 3. FEATURE BUILD, strictly point-in-time (no look-ahead)
# ============================================================================
def build_features(d: pd.DataFrame) -> pd.DataFrame:
    """
    For every commodity-month t, build features using ONLY data up to and
    including t. Volatility is an EXPANDING-WINDOW GARCH refit (not a single
    full-sample fit), and the seasonal index is re-estimated on data up to t,
    so nothing in the feature set has seen the future.
    """
    fx = implied_fx()
    fx_m3 = fx.pct_change(3) * 100
    fx_m12 = fx.pct_change(12) * 100
    fx_v12 = (np.log(fx).diff() * 100).rolling(12).std() * np.sqrt(12)

    # market-wide 3-month momentum: the average across all other commodities,
    # capturing economy-wide food inflation rather than a single crop's story
    wide = d.pivot_table(index="month", columns="item", values="price_real_kg")
    mkt = (wide.pct_change(3) * 100).mean(axis=1)

    rows = []
    for item, g in d.groupby("item"):
        s = series_for(g)
        ret = np.log(s).diff().dropna() * 100
        if len(s) < MIN_TRAIN_OBS:
            continue
        for i in range(MIN_TRAIN_OBS, len(s)):
            hist = s.iloc[:i + 1]                    # data up to and including t
            t = hist.index[-1]
            rh = ret[ret.index <= t]

            # expanding-window GARCH volatility estimate at t
            try:
                res = arch_model(rh, vol="GARCH", p=1, q=1, dist="t",
                                 mean="Constant").fit(disp="off", show_warning=False)
                cv = res.conditional_volatility
                vol_ann = float(cv.iloc[-1] * np.sqrt(12))
                vol_prev = float(cv.iloc[-4] * np.sqrt(12)) if len(cv) > 4 else vol_ann
            except Exception:
                vol_ann = float(rh.tail(12).std() * np.sqrt(12))
                vol_prev = vol_ann

            seas = seasonal_index(hist)
            cur_m = t.month
            fut = [((cur_m - 1 + k) % 12) + 1 for k in range(1, HORIZON + 1)]
            seasonal_risk = float(np.mean([seas[f] for f in fut]) - seas[cur_m])

            ma12 = hist.tail(12).mean(); sd12 = hist.tail(12).std()
            win = hist.tail(12)
            rows.append({
                "item": item, "month": t,
                "price_real_kg": float(hist.iloc[-1]),
                "mom_3m": float(hist.iloc[-1] / hist.iloc[-4] - 1) * 100 if i >= 3 else np.nan,
                "mom_6m": float(hist.iloc[-1] / hist.iloc[-7] - 1) * 100 if i >= 6 else np.nan,
                "mom_12m": float(hist.iloc[-1] / hist.iloc[-13] - 1) * 100 if i >= 12 else np.nan,
                "trend_gap": float(hist.iloc[-1] / ma12 - 1) * 100,
                "z12": float((hist.iloc[-1] - ma12) / sd12) if sd12 > 0 else 0.0,
                "vol_ann": vol_ann,
                "vol_chg": vol_ann - vol_prev,
                "seasonal_risk": seasonal_risk,
                "pos_12m": float((hist.iloc[-1] - win.min()) /
                                 (win.max() - win.min())) if win.max() > win.min() else .5,
                "sin_m": float(np.sin(2 * np.pi * cur_m / 12)),
                "cos_m": float(np.cos(2 * np.pi * cur_m / 12)),
                "fx_mom_3m": float(fx_m3.get(t, np.nan)),
                "fx_mom_12m": float(fx_m12.get(t, np.nan)),
                "fx_vol_12m": float(fx_v12.get(t, np.nan)),
                "mkt_mom_3m": float(mkt.get(t, np.nan)),
                "rel_strength": (float(hist.iloc[-1] / hist.iloc[-4] - 1) * 100
                                 - float(mkt.get(t, 0.0))) if i >= 3 else np.nan,
            })
    f = pd.DataFrame(rows).dropna(subset=FEATURES).reset_index(drop=True)

    # label: real price rises more than the threshold over the next HORIZON months
    lab = []
    for item, g in f.groupby("item"):
        g = g.sort_values("month").copy()
        g["future_ret"] = g.price_real_kg.shift(-HORIZON) / g.price_real_kg - 1
        g["event"] = (g.future_ret > EVENT_THRESHOLD).astype(float)
        lab.append(g)
    return pd.concat(lab, ignore_index=True)


# ============================================================================
# 4. TWO MODELS: the naive composite, and the learned walk-forward model
# ============================================================================
def naive_score(f: pd.DataFrame) -> pd.Series:
    """Original equal-weight composite. Kept as the documented baseline."""
    parts = []
    for item, g in f.groupby("item"):
        g = g.copy()
        for c in ["mom_3m", "vol_ann", "trend_gap", "seasonal_risk"]:
            g[c + "_p"] = g[c].rank(pct=True) * 100
        g["naive_score"] = g[[c + "_p" for c in
                              ["mom_3m", "vol_ann", "trend_gap", "seasonal_risk"]]].mean(axis=1)
        parts.append(g[["item", "month", "naive_score"]])
    return f.merge(pd.concat(parts), on=["item", "month"], how="left")["naive_score"]


def walkforward(f: pd.DataFrame) -> pd.DataFrame:
    """
    Expanding-window walk-forward validation.
    At each prediction month T, the training set is every observation whose
    outcome window has already CLOSED (month <= T - HORIZON months). That is
    the information a real forecaster would really have had at T.
    """
    f = f.sort_values("month").reset_index(drop=True)
    months = sorted(f[f.month >= WALKFORWARD_START].month.unique())
    preds = []
    for T in months:
        train = f[(f.month <= T - pd.DateOffset(months=HORIZON))].dropna(subset=["event"])
        test = f[f.month == T]
        if len(train) < 120 or train.event.nunique() < 2 or test.empty:
            continue
        sc = StandardScaler().fit(train[FEATURES])
        m = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
        m.fit(sc.transform(train[FEATURES]), train.event)
        p = m.predict_proba(sc.transform(test[FEATURES]))[:, 1]
        t2 = test.copy()
        t2["prob"] = p
        t2["n_train"] = len(train)
        preds.append(t2)
    return pd.concat(preds, ignore_index=True)


def evaluate(df: pd.DataFrame, score_col: str, label="") -> dict:
    d = df.dropna(subset=[score_col, "event"])
    auc = float(roc_auc_score(d.event, d[score_col]))
    base = float(d.event.mean())
    q = pd.qcut(d[score_col], 5, labels=False, duplicates="drop") + 1
    tab = (d.assign(q=q).groupby("q")
             .agg(n=("event", "size"), hit_rate=("event", "mean"),
                  mean_future_ret=("future_ret", "mean")).reset_index())
    tab["hit_rate"] *= 100; tab["mean_future_ret"] *= 100
    top = tab.iloc[-1]
    print(f"    {label:<22} AUC={auc:.3f}  base={base*100:.1f}%  "
          f"top-quintile hit={top.hit_rate:.1f}%  lift={top.hit_rate/100/base:.2f}x")
    return {"auc": auc, "base_rate_pct": base * 100, "n": int(len(d)),
            "top_quintile_hit_rate_pct": float(top.hit_rate),
            "lift_top_quintile": float(top.hit_rate / 100 / base),
            "quintiles": tab.round(2).to_dict("records")}


# ============================================================================
# 5. FIGURES
# ============================================================================
def fig_prices(d: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.5))
    ax = axes[0]
    for item, g in d.groupby("item"):
        ax.plot(g.month, g.price_real_kg / g.price_real_kg.iloc[0] * 100, lw=1.5, label=item)
    ax.set_title("Real prices (constant Jan-2020 naira), indexed to first observation", loc="left")
    ax.set_ylabel("index"); ax.legend(fontsize=7.5, ncol=2, frameon=False)
    ax = axes[1]
    for item, g in d.groupby("item"):
        ax.plot(g.month, g.price_ngn_kg, lw=1.5)
    ax.set_yscale("log"); ax.set_ylabel("NGN / kg (log scale)")
    ax.set_title("Nominal naira prices", loc="left")
    fig.suptitle("Nigerian staple food prices, WFP market data, 2014-2026",
                 x=.012, ha="left", fontsize=14, weight="bold")
    save(fig, "01_price_series.png")


def fig_volatility(v, params):
    items = sorted(v.item.unique()); n = len(items)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13.5, 2.6 * rows), sharex=True)
    for ax, item in zip(axes.ravel(), items):
        g = v[v.item == item]; p = params[item]
        ax.plot(g.month, g.cond_vol_ann_pct, color=INK, lw=1.3)
        ax.axhline(p["calm_threshold"], color=GOLD, lw=.9, ls="--")
        ax.axhline(p["high_threshold"], color=RED, lw=.9, ls="--")
        ax.fill_between(g.month, 0, g.cond_vol_ann_pct,
                        where=g.cond_vol_ann_pct >= p["high_threshold"], color=RED, alpha=.30)
        ax.fill_between(g.month, 0, g.cond_vol_ann_pct,
                        where=(g.cond_vol_ann_pct < p["high_threshold"]) &
                              (g.cond_vol_ann_pct >= p["calm_threshold"]), color=AMBER, alpha=.25)
        ax.set_title(f"{item}  (persistence {p['persistence']:.2f})", fontsize=9.5, loc="left")
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Conditional volatility of real food prices, GARCH(1,1) with Student-t errors\n"
                 "Amber = elevated regime (>60th pct), red = high-risk regime (>85th pct)",
                 x=.012, ha="left", fontsize=14, weight="bold")
    save(fig, "02_volatility_regimes.png")


def fig_seasonality(s):
    piv = s.pivot(index="item", columns="month_num", values="seasonal_pct")
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    vmax = np.nanmax(np.abs(piv.values))
    im = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(12)); ax.set_xticklabels(names)
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            ax.text(j, i, f"{val:+.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(val) > vmax*.55 else INK)
    ax.set_title("Seasonal price signature by commodity\n"
                 "% deviation from annual average after removing trend, the lean-season pattern",
                 loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=.8, label="% vs annual average")
    save(fig, "03_seasonality.png")


def fig_backtest(naive: dict, learned: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.3))
    ax = axes[0]
    for res, col, lab in [(naive, GREY, f"Naive equal-weight (AUC {naive['auc']:.2f})"),
                          (learned, GREEN, f"Learned walk-forward (AUC {learned['auc']:.2f})")]:
        q = pd.DataFrame(res["quintiles"])
        ax.plot(q.q, q.hit_rate, "o-", color=col, lw=2.4, ms=7, label=lab)
    ax.axhline(learned["base_rate_pct"], color=RED, ls="--", lw=1.4,
               label=f"base rate ({learned['base_rate_pct']:.0f}%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Risk-score quintile (5 = highest predicted risk)")
    ax.set_ylabel(f"% followed by a >{EVENT_THRESHOLD*100:.0f}% real price rise")
    ax.set_title("Hit rate by predicted-risk quintile", loc="left")
    ax.legend(frameon=False, fontsize=8.5)

    ax = axes[1]
    names = ["Naive\nequal-weight", "Learned\nwalk-forward"]
    vals = [naive["auc"], learned["auc"]]
    ax.bar(names, vals, color=[GREY, GREEN], width=.5)
    ax.axhline(.5, color=RED, ls="--", lw=1.4)
    ax.text(1.52, .505, "no skill", color=RED, fontsize=9, ha="right")
    ax.set_ylim(.4, max(vals) * 1.18); ax.set_ylabel("ROC-AUC")
    for i, v in enumerate(vals):
        ax.text(i, v + .006, f"{v:.3f}", ha="center", fontsize=12, weight="bold")
    ax.set_title("Out-of-sample discrimination", loc="left")
    fig.suptitle("Validating the early-warning signal: what failed, and what fixed it",
                 x=.012, ha="left", fontsize=14, weight="bold")
    save(fig, "04_backtest.png")


def fig_coefficients(f: pd.DataFrame):
    """Direction and strength of each feature in the final full-sample fit."""
    d = f.dropna(subset=["event"])
    sc = StandardScaler().fit(d[FEATURES])
    m = LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced")
    m.fit(sc.transform(d[FEATURES]), d.event)
    co = pd.Series(m.coef_[0], index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    ax.barh(co.index, co.values, color=[RED if v < 0 else GREEN for v in co.values])
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("Standardised logistic coefficient (log-odds of a price shock)")
    ax.set_title("What the model learned\n"
                 "Negative = mean reversion: a high recent price makes a further rise LESS likely",
                 loc="left")
    save(fig, "05_model_coefficients.png")
    return co.round(3).to_dict()


def forecast_all(d: pd.DataFrame, steps=6) -> pd.DataFrame:
    rows = []
    for item, g in d.groupby("item"):
        s = series_for(g)
        if len(s) < 48:
            continue
        y = np.log(s)
        try:
            m = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            fc = m.get_forecast(steps); ci = fc.conf_int(alpha=.20)
            idx = pd.date_range(y.index[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS")
            for i, dt in enumerate(idx):
                rows.append({"item": item, "month": dt,
                             "forecast_real_kg": float(np.exp(fc.predicted_mean.iloc[i])),
                             "lo80": float(np.exp(ci.iloc[i, 0])),
                             "hi80": float(np.exp(ci.iloc[i, 1]))})
        except Exception as ex:
            print(f"    forecast failed for {item}: {ex}")
    return pd.DataFrame(rows)


# ============================================================================
def main():
    d = load()
    print(f"Panel: {len(d):,} commodity-months, {d.item.nunique()} commodities, "
          f"{d.month.min().date()} to {d.month.max().date()}")
    OUT["panel"] = {"rows": int(len(d)), "commodities": int(d.item.nunique()),
                    "start": str(d.month.min().date()), "end": str(d.month.max().date())}
    fig_prices(d)

    print("\n1) GARCH(1,1)-t volatility regimes (descriptive, full sample)")
    v, params = garch_by_commodity(d)
    fig_volatility(v, params)
    OUT["garch"] = params

    print("\n2) Seasonality")
    seas = seasonality_table(d)
    seas.to_csv(TAB / "seasonality.csv", index=False)
    fig_seasonality(seas)
    peak = seas.loc[seas.groupby("item").seasonal_pct.idxmax()].set_index("item")
    OUT["seasonal_peak_month"] = peak[["month_num", "seasonal_pct"]].round(2).to_dict("index")
    print(peak[["month_num", "seasonal_pct"]].round(1).to_string())

    print("\n3) Point-in-time feature build (expanding-window GARCH + seasonality)")
    f = build_features(d)
    f["naive_score"] = naive_score(f)
    f.to_csv(TAB / "features.csv", index=False)
    print(f"    {len(f):,} feature rows, {f.dropna(subset=['event']).event.mean()*100:.1f}% "
          f"are followed by a >{EVENT_THRESHOLD*100:.0f}% real price rise")

    print("\n4) Model comparison")
    naive_res = evaluate(f.dropna(subset=["event"]), "naive_score", "naive equal-weight")
    wf = walkforward(f)
    wf.to_csv(TAB / "walkforward_predictions.csv", index=False)
    learned_res = evaluate(wf, "prob", "learned walk-forward")
    fig_backtest(naive_res, learned_res)
    OUT["backtest"] = {
        "event_definition": f">{EVENT_THRESHOLD*100:.0f}% real price rise over {HORIZON} months",
        "naive_equal_weight": naive_res,
        "learned_walkforward": learned_res,
        "walkforward_start": WALKFORWARD_START,
        "n_walkforward_predictions": int(len(wf)),
    }
    OUT["coefficients"] = fig_coefficients(f)
    print("    learned coefficients:", OUT["coefficients"])

    print("\n5) Current status")
    last = wf.sort_values("month").groupby("item").tail(1)
    reg = v.sort_values("month").groupby("item").tail(1).set_index("item")["regime"]
    cur = last.set_index("item")[["month", "price_real_kg", "prob", "mom_3m",
                                  "vol_ann", "seasonal_risk"]].copy()
    cur["regime"] = reg
    cur["risk_level"] = pd.cut(cur.prob, [-np.inf, .25, .45, .65, np.inf],
                               labels=["Low", "Watch", "Elevated", "Alert"]).astype(str)
    cur = cur.sort_values("prob", ascending=False)
    cur.to_csv(TAB / "current_status.csv")
    OUT["current_status"] = json.loads(cur.reset_index().to_json(orient="records",
                                                                 date_format="iso"))
    print(cur.round(3).to_string())

    print("\n6) SARIMA forecasts")
    fc = forecast_all(d)
    fc.to_csv(TAB / "forecasts.csv", index=False)
    print(f"    {len(fc)} rows for {fc.item.nunique()} commodities")

    with open(ROOT / "outputs" / "model_results.json", "w") as fh:
        json.dump(OUT, fh, indent=2, default=str)
    print("\nSaved outputs/model_results.json")


if __name__ == "__main__":
    main()
