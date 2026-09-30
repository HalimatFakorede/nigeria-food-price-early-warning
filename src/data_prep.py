"""
Project 2, Nigeria Food Price Early Warning System
Step 1: Data collection, unit normalisation and panel construction.

Source (open, no authentication):
  * WFP / HDX "Nigeria - Food Prices", market-level retail & wholesale prices,
    2002 to present, 68 markets across 14 states, NGN and USD.
  * World Bank WDI, Nigeria CPI, used to convert nominal naira prices to real
    (inflation-adjusted) prices.

THE CENTRAL DATA PROBLEM, AND HOW IT IS SOLVED
  Prices are quoted in inconsistent units: "100 KG", "2.5 KG", "400 G", "L", "KG".
  A raw price column is therefore meaningless to compare across rows. Every price
  is normalised to NGN per kilogramme (or per litre for liquids) before anything
  else happens. Getting this wrong silently destroys the entire analysis, so the
  parser is explicit and anything it cannot parse is dropped and counted.

Output: data/processed/*.csv
Run:  python src/data_prep.py
"""
from __future__ import annotations
import pathlib, re
import numpy as np
import pandas as pd
import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
for d in (RAW, PROC):
    d.mkdir(parents=True, exist_ok=True)

HDX_PRICES = ("https://data.humdata.org/dataset/42db041f-7aaf-4ab4-961f-2a12096861e7/"
              "resource/12b51155-0cd3-4806-9924-61ede4077591/download/wfp_food_prices_nga.csv")

# Staples that matter for Nigerian household food security.
FOCUS = [
    "Maize (white)", "Maize (yellow)", "Rice (local)", "Rice (imported)",
    "Millet", "Sorghum", "Sorghum (white)", "Sorghum (brown)",
    "Yam", "Gari (white)", "Gari (yellow)", "Cassava meal (gari)",
    "Cowpeas (white)", "Cowpeas (brown)", "Beans (white)", "Beans (red)",
    "Groundnuts", "Groundnuts (shelled)", "Oil (palm)", "Oil (vegetable)",
]

# Collapse near-duplicate WFP labels into analysis commodities
COMMODITY_MAP = {
    "Maize (white)": "Maize", "Maize (yellow)": "Maize",
    "Rice (local)": "Rice (local)", "Rice (imported)": "Rice (imported)",
    "Millet": "Millet",
    "Sorghum": "Sorghum", "Sorghum (white)": "Sorghum", "Sorghum (brown)": "Sorghum",
    "Yam": "Yam",
    "Gari (white)": "Gari", "Gari (yellow)": "Gari", "Cassava meal (gari)": "Gari",
    "Cowpeas (white)": "Cowpea", "Cowpeas (brown)": "Cowpea",
    "Beans (white)": "Beans", "Beans (red)": "Beans",
    "Groundnuts": "Groundnut", "Groundnuts (shelled)": "Groundnut",
    "Oil (palm)": "Palm oil", "Oil (vegetable)": "Vegetable oil",
}

MIN_MONTHS = 100         # a live series must have >= ~8 years of monthly observations
MIN_MARKETS_PER_MONTH = 3    # thin months (1-2 markets) are not a national price
MIN_MONTHS_LONG = 150    # long-history (wholesale) series used for crisis context


# ---------------------------------------------------------------------------
def parse_unit(u: str) -> float | None:
    """
    Convert a WFP unit string to kilogrammes (litres treated as 1 unit).
    '100 KG' -> 100 | '400 G' -> 0.4 | 'KG' -> 1 | 'L' -> 1 | '2.5 KG' -> 2.5
    Returns None for units that cannot be mass-normalised (e.g. 'Head', 'Unit').
    """
    if not isinstance(u, str):
        return None
    s = u.strip().upper()
    m = re.match(r"^([\d.]+)?\s*(KG|G|L|ML|MT)$", s)
    if not m:
        return None
    qty = float(m.group(1)) if m.group(1) else 1.0
    unit = m.group(2)
    factor = {"KG": 1.0, "G": 0.001, "L": 1.0, "ML": 0.001, "MT": 1000.0}[unit]
    return qty * factor


def load_raw() -> pd.DataFrame:
    path = RAW / "wfp_food_prices_nga.csv"
    if not path.exists():
        print("  downloading WFP Nigeria food prices from HDX ...")
        r = requests.get(HDX_PRICES, timeout=300); r.raise_for_status()
        path.write_bytes(r.content)
    df = pd.read_csv(path, low_memory=False)
    df = df[df["date"] != "#date"]                       # strip the HXL tag row
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["price", "usdprice", "latitude", "longitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date", "price"])


def fetch_cpi() -> pd.DataFrame:
    """Nigeria consumer price index (2010=100), annual, interpolated to monthly."""
    out = RAW / "nigeria_cpi.csv"
    if out.exists():
        return pd.read_csv(out, parse_dates=["month"])
    url = ("https://api.worldbank.org/v2/country/NGA/indicator/FP.CPI.TOTL"
           "?format=json&per_page=500&date=2000:2026")
    j = requests.get(url, timeout=90).json()
    a = pd.DataFrame([{"year": int(x["date"]), "cpi": x["value"]} for x in j[1]])
    a = a.dropna().sort_values("year")
    # extend forward to cover months beyond the last annual observation
    idx = pd.date_range(f"{a.year.min()}-01-01", "2026-12-01", freq="MS")
    m = pd.DataFrame({"month": idx})
    m["year"] = m.month.dt.year
    m = m.merge(a, on="year", how="left")
    # place annual value at mid-year, interpolate, then extrapolate the tail
    m.loc[m.month.dt.month != 7, "cpi"] = np.nan
    m["cpi"] = m["cpi"].interpolate(limit_direction="both")
    last_growth = a.cpi.pct_change().tail(3).mean()
    for i in range(len(m)):
        if pd.isna(m.loc[i, "cpi"]) and i > 0:
            m.loc[i, "cpi"] = m.loc[i - 1, "cpi"] * (1 + last_growth / 12)
    m[["month", "cpi"]].to_csv(out, index=False)
    return m[["month", "cpi"]]


# ---------------------------------------------------------------------------
def main() -> None:
    print("STEP 1  load raw WFP market prices")
    df = load_raw()
    n0 = len(df)
    print(f"        {n0:,} raw observations, {df.date.min().date()} to {df.date.max().date()}")

    print("STEP 2  filter to staple food commodities")
    df = df[df.commodity.isin(FOCUS)].copy()
    df["item"] = df.commodity.map(COMMODITY_MAP)
    print(f"        {len(df):,} rows across {df.item.nunique()} analysis commodities")

    print("STEP 3  normalise units to price per kg / litre")
    df["unit_kg"] = df.unit.map(parse_unit)
    bad = df.unit_kg.isna().sum()
    df = df.dropna(subset=["unit_kg"])
    df = df[df.unit_kg > 0]
    df["price_per_kg"] = df.price / df.unit_kg
    df["usd_per_kg"] = df.usdprice / df.unit_kg
    print(f"        dropped {bad:,} rows with non-mass units "
          f"({bad/max(n0,1)*100:.1f}% of raw); {len(df):,} remain")

    print("STEP 4  outlier control")
    # Winsorise within commodity x pricetype at the 1st/99th percentile.
    # Market price data contains data-entry errors of 10-100x; a single one of
    # those will dominate a volatility model if left in.
    before = len(df)
    def clip(g):
        lo, hi = g.price_per_kg.quantile([.01, .99])
        return g[g.price_per_kg.between(lo, hi)]
    df = (df.groupby(["item", "pricetype"], group_keys=False)[df.columns.tolist()]
            .apply(clip))
    print(f"        removed {before-len(df):,} extreme values (1st/99th pct within commodity)")

    df["month"] = df.date.values.astype("datetime64[M]")

    print("STEP 5  build monthly national panel (median across markets)")
    nat = (df[df.pricetype == "Retail"]
           .groupby(["item", "month"])
           .agg(price_ngn_kg=("price_per_kg", "median"),
                price_usd_kg=("usd_per_kg", "median"),
                n_markets=("market", "nunique"),
                n_obs=("price_per_kg", "size"))
           .reset_index())
    nat = nat[nat.n_markets >= MIN_MARKETS_PER_MONTH]
    counts = nat.groupby("item").size()
    keep = counts[counts >= MIN_MONTHS].index
    nat = nat[nat.item.isin(keep)]
    print(f"        kept {len(keep)} commodities with >= {MIN_MONTHS} months: {sorted(keep)}")

    print("STEP 6  deflate to real prices")
    cpi = fetch_cpi()
    base = cpi[cpi.month == "2020-01-01"].cpi.values[0]
    nat = nat.merge(cpi, on="month", how="left")
    nat["price_real_kg"] = nat.price_ngn_kg / nat.cpi * base      # constant Jan-2020 naira
    nat = nat.sort_values(["item", "month"]).reset_index(drop=True)

    print("STEP 7  build long-history wholesale panel (crisis context)")
    # Retail collection only begins in 2014. WFP wholesale series reach back to
    # 2002 for several cereals but were discontinued in 2023. Keeping them as a
    # SEPARATE panel lets the analysis show the 2008 and 2016 price crises
    # without silently splicing two different price concepts into one series.
    who = (df[df.pricetype == "Wholesale"]
           .groupby(["item", "month"])
           .agg(price_ngn_kg=("price_per_kg", "median"),
                n_markets=("market", "nunique"))
           .reset_index())
    cnt = who.groupby("item").size()
    who = who[who.item.isin(cnt[cnt >= MIN_MONTHS_LONG].index)]
    who = who.merge(cpi, on="month", how="left")
    who["price_real_kg"] = who.price_ngn_kg / who.cpi * base
    who = who.sort_values(["item", "month"])
    print(f"        {who.item.nunique()} long series: {sorted(who.item.unique())} "
          f"({who.month.min().date()} to {who.month.max().date()})")
    who.to_csv(PROC / "wholesale_long_monthly.csv", index=False)

    print("STEP 8  build state-level panel")
    state = (df[df.pricetype == "Retail"]
             .groupby(["item", "admin1", "month"])
             .agg(price_ngn_kg=("price_per_kg", "median"),
                  n_markets=("market", "nunique"))
             .reset_index()
             .rename(columns={"admin1": "state"}))
    state = state.merge(cpi, on="month", how="left")
    state["price_real_kg"] = state.price_ngn_kg / state.cpi * base

    mkt = (df.groupby(["market", "admin1"])
             .agg(lat=("latitude", "first"), lon=("longitude", "first"),
                  n=("price_per_kg", "size")).reset_index())

    nat.to_csv(PROC / "national_monthly.csv", index=False)
    state.to_csv(PROC / "state_monthly.csv", index=False)
    mkt.to_csv(PROC / "markets.csv", index=False)
    df.to_csv(PROC / "clean_observations.csv.gz", index=False, compression="gzip")

    print("\nSaved:")
    for f in ["national_monthly.csv", "wholesale_long_monthly.csv",
              "state_monthly.csv", "markets.csv", "clean_observations.csv.gz"]:
        print(f"  data/processed/{f}")
    print(f"\nNational panel: {len(nat):,} commodity-months, "
          f"{nat.month.min().date()} to {nat.month.max().date()}")
    print(nat.groupby("item").agg(months=("month", "size"),
                                  first=("month", "min"),
                                  last=("month", "max")).to_string())


if __name__ == "__main__":
    main()
