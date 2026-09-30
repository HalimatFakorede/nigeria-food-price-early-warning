# Findings Brief, What 12 Years of Nigerian Market Prices Say About Food Price Risk

**Halimat H. Fakorede** | WFP market price data, 9 staples, 68 markets, 14 states, 2014-2026

---

## Summary

Nigerian staple food prices are highly volatile, strongly seasonal, and driven more by the exchange rate than by anything happening in the food system itself. A shock-risk model validated out-of-sample identifies months that precede a >10% real price rise at **1.75× the base rate**, useful for triage and procurement timing, but not a precise forecast.

---

## 1. Naira depreciation is the dominant driver

Of sixteen features tested, **12-month FX depreciation has the largest standardized coefficient (+1.07)**, larger than every price-based technical indicator combined.

For imported rice and vegetable oil this is direct pass-through. For domestic cereals it operates through imported fertilizer, diesel for transport and milling, and agrochemicals. The 2023 float and 2024 devaluation, from ₦685 to ₦1,566 per dollar, propagated through every staple in the basket.

**Implication:** a Nigerian food-security monitoring system that tracks only food prices is monitoring a lagging indicator. FX should sit on the front page.

## 2. Prices mean-revert over three months

Momentum correlates **−0.19** with the next three months' real return; the deviation from the 12-month average correlates **−0.09**. A staple that has just spiked is more likely to fall back than to keep climbing.

**Implication:** reacting to a spike after it has happened is usually reacting too late, and intervention designs that assume a spike will persist for a quarter will systematically over-provision. Buffer releases timed to the *seasonal* calendar beat buffer releases timed to *recent price moves*.

## 3. The lean season is large, predictable, and ignored

| Commodity | Peak month | Premium over annual average |
|---|---|---|
| Millet | August | +15.8% |
| Yam | July | +15.2% |
| Sorghum | August | +13.8% |
| Beans | August | +11.5% |
| Groundnut | August | +10.6% |

Cereals peak in **August**, immediately before the main harvest, household stocks exhausted, new crop not yet in.

**Implication:** this is the single most actionable pattern in the dataset. It is large, it recurs annually, and it is forecastable a year ahead. Procurement in the lean season is the most expensive possible timing. Cash transfers of a fixed naira value lose 10-16% of their real purchasing power at exactly the moment they matter most, unless indexed.

## 4. Volatility is persistent, turbulence does not fade

GARCH(1,1) persistence (α+β) is **at or above 0.95 for seven of nine commodities**, and effectively 1.00 for imported rice, local rice, palm oil and vegetable oil.

Practically: once a market becomes turbulent, it stays turbulent. There is no rapid reversion to calm. Imported rice is the most volatile staple at roughly **70% annualised**, the commodity most exposed to FX.

**Implication:** a volatility spike is not a transient to be waited out. And for the four near-IGARCH series, long-horizon volatility forecasts are not statistically supportable and should not be published.

## 5. The warning signal works, modestly, and honestly

| | Naive equal-weight | Learnt, walk-forward |
|---|---|---|
| ROC-AUC | 0.583 | **0.608** |
| Top-quintile hit rate | 33.5% | **50.0%** |
| Lift vs 28.6% base rate | 1.28× | **1.75×** |

Months in the highest-risk quintile are followed by a >10% real price rise half the time, against a base rate of 28.6%.

**Use it for:** monthly triage across commodities, procurement timing, cash-transfer indexing.
**Do not use it for:** point price forecasts, or as the sole trigger for a food-security declaration.

---

## Recommended next additions

1. **Monthly NBS CPI** instead of interpolated annual, sharper real prices.
2. **Rainfall (CHIRPS) and NDVI (Sentinel-2)**, the supply side is entirely absent from the current model.
3. **ACLED conflict data**, insecurity disrupts both markets and the price collection itself, and the resulting gaps are missing-not-at-random.
4. **State-level models**, a national average hides the state where a crisis begins.

---

*Full method, code, backtest and limitations: [github.com/HalimatFakorede](https://github.com/HalimatFakorede)*
