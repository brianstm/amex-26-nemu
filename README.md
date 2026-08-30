# NEMU

**Notice · Explain · Match · Uplift**

Proof of concept for American Express: estimate cross-border spend an issuer is losing to competitors, explain why, and recommend a fix.

The ledger is **hybrid**. No American Express or card-member data was used. Trips, spend, and acceptance are simulated so Notice has a known causal structure. Ticket fields a real issuer file would also carry — currency, FX, MCC, merchant name — come from public sources (ISO 4217, World Bank, ISO 18245, OpenStreetMap). Country knobs (prices, cash, trip volume) are calibrated to Findex, PPP, and OSM counts.

| Hidden (never in the likelihood) | Model found | Cause $ accuracy | Corridor corr. | Holdout calibration |
| ---: | ---: | ---: | ---: | ---: |
| **USD 7.34M** acceptance leakage | **USD 6.82M (93%)** | **92%** | 0.964 | 1.01 |

Seed `26`. Knobs from `data/external/calibration.json`. Scored against `outputs/ground_truth.csv`, which is not in the PPML likelihood.

---

## Quick start

Python **3.12**. Use the project venv — Homebrew Python 3.10 ships a SciPy that fails to `dlopen` on recent macOS.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_pipeline.py
.venv/bin/streamlit run dashboard/app.py
```

Walkthrough: [`notebooks/demo_walkthrough.ipynb`](notebooks/demo_walkthrough.ipynb).

Optional refresh of public knobs (a snapshot is already committed):

```bash
.venv/bin/python -m data.calibrate_external          # World Bank + OSM amenity counts
.venv/bin/python -m data.calibrate_external --skip-osm
.venv/bin/python -m data.fetch_ticket_features       # ISO FX + OSM merchant names
```

---

## Pipeline

```mermaid
flowchart LR
  A[pandas / numpy DGP] --> B[statsmodels PPML]
  B --> C[sklearn PU classifier]
  C --> D[econml CausalForest]
  D --> E[Streamlit + Plotly]
```

```
generate → Notice (PPML) → Explain (cause) → Match (merchants + offers) → Uplift (holdout)
```

1. **Notice** — reconstruct presence; estimate counterfactual billed business at near-saturated acceptance.
2. **Explain** — tag each residual as `no_acceptance`, `rail_substitution`, `cash`, or `no_demand`.
3. **Match** — acquiring for coverage holes; smallest profitable offer for rail substitution.
4. **Uplift** — randomized holdout; count only incremental billed business versus control.

| Layer | Tools | Role |
|---|---|---|
| Runtime | Python 3.12, `venv` | Whole repo |
| Tables | pandas, numpy | Ledger, trip × category panel |
| Notice | statsmodels Poisson GLM (PPML, HC0) | Counterfactual billed business at $a = 0.95$ |
| Explain | scikit-learn logistic + `OneHotEncoder` | Elkan–Noto PU stretch on declined auths |
| Match — acquiring | numpy greedy submodular | Targeting under coverage $1 - e^{-\lambda n}$ |
| Match — offers | econml `CausalForestDML` (X-learner fallback) | Heterogeneous effects for `2x_points` / `statement_credit` |
| Uplift | numpy bootstrap | Holdout calibration of realized vs predicted $\Delta$ |
| Dashboard | Streamlit, Plotly | `dashboard/app.py` |
| Calibration | World Bank API, Nominatim, Overpass; `certifi` | `data/calibrate_external.py` |

econml is optional at runtime: if `CausalForestDML` fails to import or fit, Match falls back to the X-learner.

---

## Data

`data/generate_synthetic.py` draws **5,000 members** and **15,000 trips**. Spend and acceptance are simulated. Public ticket fields are attached from `data/external/real_features.json`.

| File | What it is | Who sees it |
|---|---|---|
| `outputs/members.csv` | Home market, segment, domestic category intensity | Model |
| `outputs/trips.csv` | Destination, dates, trip type, acceptance, cash, ISO currency, FX | Model |
| `outputs/transactions.csv` | Captured tickets: USD amount, local amount, currency, MCC, OSM merchant name | Model |
| `outputs/declines.csv` | Simulated “card not accepted” auths (PU positives) | Explain only |
| `outputs/ground_truth.csv` | True spend, leakage, oracle cause | **Validation only.** Never in the likelihood |

### Ticket fields: real vs simulated

| Field | Source | Real or simulated |
|---|---|---|
| `currency` | ISO 4217 | Real code |
| `fx_lcu_per_usd` / `amount_local` | World Bank `PA.NUS.FCRF` | Real rate; USD amount is simulated |
| `mcc` | ISO 18245 (5812 dining, 5311 retail, 4111 transport, 7011 lodging) | Real code |
| `merchant_name` | OSM POI names (ODbL), else `data/public_venues.py` or a public operator | Real name; *which ticket* gets it is simulated |
| `amount` / `amount_usd` | DGP | Simulated |
| trip, nights, `acceptance_density` | DGP | Simulated (identifying variation) |
| `cash_intensity` | Global Findex | Real country statistic |

Pitch line: *the causal structure is simulated so we can score Notice; currency, FX, MCC, and merchant strings are public data an issuer file would also contain.*

---

## Why not Kaggle fraud files

Two public “credit card transaction” dumps were considered and **not** swapped in. Both are still synthetic, and both answer `is_fraud`, not coverage leakage.

| | [Jangra 500k](https://www.kaggle.com/datasets/kuldeepjangra/credit-card-fraud-detection-500k-transactions) | [Choksi 1.85M](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset/data) |
|---|---|---|
| What it is | Author-labelled synthetic fraud set | Sparkov simulator (same family as `kartik2112/fraud-detection`) |
| Target | `is_fraud` | `is_fraud` |
| Geography | Simulated cities | US cards; no APAC trip corridors |
| Amount | One observed ticket | One observed ticket (`amt`) |

Notice needs `acceptance_density`, `cash_intensity`, a trip × district grid (including zeros), hidden `true_spend`, and an oracle cause. Neither file has those. Feeding them into PPML would estimate $E[\text{amount} \mid x]$ for **captured** tickets and call the residual “leakage.”

> **Stolen PAN dumps.** A file with live PANs, CVVs, or “fullz” is stolen card data. Do not download it. The two Kaggle sets above are labelled simulators — they are the wrong estimand, not illegal.

---

## Country calibration

Row-level PANs and tickets are not public. Country statistics are. The generator stays synthetic at the row and borrows four public series for the knobs.

### Price — World Bank PPP / PLI

[`PA.NUS.GDP.PLI`](https://data.worldbank.org/indicator/PA.NUS.GDP.PLI) (GDP price level index, US $= 100$). Older code `PA.NUS.PPPC.RF` is archived.

$$
P_c = \frac{\mathrm{PLI}_c}{\mathrm{PLI}_{\mathrm{SG}}}
$$

Singapore $= 1$. Destination-country fixed effects later absorb $P_c$; it is here so *levels* of spend look like a tourist-price map.

### Cash — Global Findex 2025

In-store items (`fin25e2` / `fin25e2b`) exist for Indonesia, Malaysia, Thailand, Vietnam. Elsewhere: `g20.made` (made a digital payment, % age $15+$).

$$
c_c =
\begin{cases}
\dfrac{\texttt{fin25e2b}_c}{\texttt{fin25e2}_c + \texttt{fin25e2b}_c}
  & \text{in-store items present} \\[0.8em]
1 - \texttt{g20.made}_c / 100
  & \text{otherwise}
\end{cases}
$$

Clipped to $[0.04, 0.85]$. Snapshot in `data/external/calibration.json` (fetched 2026-08-24):

| Destination | $P_c$ (SG $=1$) | Cash $c_c$ | Findex source |
|---|---:|---:|---|
| Australia | 1.50 | 0.04 | $1$ minus digital payment (2021) |
| France | 1.27 | 0.04 | $1$ minus digital payment (2021) |
| Hong Kong | 1.17 | 0.14 | $1$ minus digital payment (2021) |
| Italy | 1.14 | 0.07 | $1$ minus digital payment (2021) |
| Japan | 1.07 | 0.12 | $1$ minus digital payment (2021) |
| Singapore | 1.00 | 0.09 | $1$ minus digital payment (2021) |
| South Korea | 0.95 | 0.04 | $1$ minus digital payment (2021) |
| Malaysia | 0.52 | 0.47 | in-store non-card share (2024) |
| Thailand | 0.51 | 0.46 | in-store non-card share (2024) |
| Indonesia | 0.47 | 0.80 | in-store non-card share (2024) |
| Vietnam | 0.46 | 0.32 | in-store non-card share (2024) |

Cash $\ge 0.30$ is the Explain threshold, so MY / TH / ID / VN are the cash-confound markets. Assigned at *country* level. District acceptance is assigned separately — that is what identifies the gravity coefficient.

### Trip volume — arrivals × OSM amenities

WDI [`ST.INT.ARVL`](https://data.worldbank.org/indicator/ST.INT.ARVL), latest year in 2015–2019. Optional OSM: Nominatim centroid, Overpass count of dining/lodging nodes within $1.2\,\mathrm{km}$. France arrivals are squashed so APAC is not swallowed:

$$
w_d = \bigl(0.55 + 0.45 \cdot r_{c(d)}^{0.4}\bigr) \cdot \frac{\log(1+n_d^{\mathrm{OSM}})}{\overline{\log(1+n^{\mathrm{OSM}})}_c}
$$

### Acceptance density is designed, not OSM

OSM counts mapped venues, not issuer terminals. A dense cash strip (Khao San) would look saturated. District $a_d$ is a **designed treatment**: cores $\ge 0.9$ (Shibuya, Orchard, Central) vs sparse $0.14$–$0.33$ (Ubud, Old Quarter, Khao San). That within-country spread is the identification strategy.

> Findex is a survey of adults, not Amex tickets. PLI is a GDP basket, not a tourist dining index. OSM is a map, not a terminal census. Currency codes and merchant *names* are public; amounts and which member spent them are simulated.

---

## Methods

### Data-generating process

$$
\text{true}_{ic} = \text{domestic}_{ic} \cdot \frac{\text{nights}}{7} \cdot P_{\text{dest}} \cdot \tau_{\text{type},c} \cdot \varepsilon_{\text{lognormal}}
$$

$$
\text{observed} = \text{true} \cdot f(a) \cdot g(c) \cdot \eta
$$

clipped to $[0, \text{true}]$, plus an extensive-margin zero when the card is never presented.

Capture is a sigmoid, not a linear share-of-wallet:

$$
f(a) = \frac{1}{1 + \exp\bigl(-k(a - m)\bigr)}, \quad k = 8,\ m = 0.40
$$

At $a = 0.2$ the issuer captures $\sim 17\%$ of willing demand; at $a = 0.9$ it captures $\sim 98\%$.

Cash confound, independent of acceptance:

$$
g(c) = 1 - c \cdot w_{\text{category}}
$$

with $w$ highest for dining ($1.00$) and lowest for lodging ($0.25$).

$$
\begin{aligned}
\text{total leakage} &= \text{true} - \text{observed} \\
\text{acceptance leakage} &= \text{true} \cdot f(0.95) \cdot g(c) - \text{observed}
\end{aligned}
$$

Notice is designed to recover **acceptance leakage**. Total leakage also includes cash; we do not claim that.

### Notice — PPML gravity

The estimation sample is the Cartesian **trip × category** grid with zeros filled in. Dropping zeros throws away the extensive margin.

Santos Silva & Tenreyro (2006): $\log y \mid y > 0$ OLS is inconsistent for $E[y \mid x]$ under zero-inflation and heteroskedasticity. Poisson pseudo-maximum-likelihood is consistent for the conditional mean. Spend in dollars is a valid PPML outcome. SEs are HC0.

$$
\begin{aligned}
\log E[\text{spend}_{ijc}]
&= \alpha_{\text{origin}} + \gamma_{\text{dest country}} + \delta_{\text{category}} \\
&\quad + \beta_1 \log a + \beta_2 (\log a)^2 \\
&\quad + \theta \log(\text{domestic intensity}) \\
&\quad + \phi_{\text{type}} + \psi \log(\text{nights}) + \lambda_{\text{segment}}
\end{aligned}
$$

$a$ varies at the *district* level. Destination-*country* FEs absorb cash, prices, and taste. $\beta_1, \beta_2$ are identified from Ubud vs SCBD and Khao San vs Sukhumvit, not from “Vietnam is poorer than Japan.”

| Acceptance term | Row corr vs hidden | Corridor corr | Recovery |
|---|---|---|---|
| $\log a$ only | 0.77 | 0.90 | 132% |
| $a + a^2$ | 0.81 | 0.97 | 82% |
| $\log a + a$ | 0.81 | 0.97 | 89% |
| **$\log a + (\log a)^2$** (chosen) | **0.81** | **0.975** | **95%** |

The quadratic lets capture flatten near saturation without baking the DGP’s $k$ and $m$ into the estimator. After Findex / PPP / OSM calibration the chosen spec recovers **93%** of hidden acceptance leakage (corridor corr $0.964$).

Estimand is the change in the **conditional mean**, not $\widehat{\text{CF}} - \text{observed}$:

$$
\widehat{\text{leakage}} = \hat E[\text{spend} \mid a = 0.95, x] - \hat E[\text{spend} \mid a = a_0, x]
$$

clipped at $0$.

### Explain — cause classifier

| Cause | Action | Rule (domain thresholds, not tuned on hidden labels) |
|---|---|---|
| `no_demand` | Do nothing | Counterfactual spend $< \$40$, or leakage $< \$10$ and observed $\approx 0$ |
| `no_acceptance` | Send acquiring | Acceptance $< 0.35$ **and** extensive margin (observed $< \$8$) |
| `cash` | Do nothing this quarter | Cash intensity $\ge 0.30$, acceptance $\ge 0.35$, material leakage |
| `rail_substitution` | Send an offer | Residual: spend happened, just not on this rail |

Priority: no demand first, then sparse extensive-margin coverage, then cash, else rail.

**PU stretch (Elkan & Noto, 2008).** Declined auths are confirmed positives for `no_acceptance`; silence is unlabeled. Fit $P(\text{labeled} \mid x)$, estimate $c = P(\text{labeled} \mid y=1)$ as the mean score on labeled positives, then $P(y=1 \mid x) \approx P(\text{labeled} \mid x) / c$. A high PU posterior can promote a row into `no_acceptance` only in the sparse extensive-margin tail. It cannot override `no_demand`.

### Match — merchants and offers

**Acquiring (`no_acceptance`).** Districts ranked by recoverable value. Coverage of a district with $n$ new merchants:

$$
\text{coverage}(n) = 1 - e^{-\lambda n}, \quad \lambda = 0.45
$$

Value of a set $S$ is $\sum_d r_d \cdot \text{coverage}(n_d(S))$. Monotone submodular, so greedy is $(1 - 1/e)$-optimal (Nemhauser, Wolsey, Fisher 1978).

**Incentives (`rail_substitution`).** Three arms: `no_offer`, `2x_points`, `statement_credit`. Training assignment is randomized. Oracle CATE (estimator does not receive it):

$$
\begin{aligned}
\tau_{\text{2x}} &= 0.05 + 0.12\cdot\mathbf{1}_{\text{dining}} + 0.06\cdot\mathbf{1}_{\text{high spend}} + 0.05\cdot\mathbf{1}_{\text{leisure}} - 0.04\cdot\mathbf{1}_{\text{retail}} \\
\tau_{\text{credit}} &= 0.06 + 0.13\cdot\mathbf{1}_{\text{retail}} + 0.07\cdot\mathbf{1}_{\text{core}} - 0.03\cdot\mathbf{1}_{\text{high spend}} + 0.02\cdot\mathbf{1}_{\text{dining}}
\end{aligned}
$$

Recovered with **CausalForestDML** or an **X-learner** (Künzel et al. 2019). Recommendation maximises net value:

$$
\text{net} = \hat\tau \cdot \text{spend} \cdot \underbrace{0.12}_{\text{take-rate}} - \text{incentive cost}
$$

If every arm has negative net, recommend `no_offer`.

### Uplift — holdout

Eligible members (recommended arm $\neq$ `no_offer`) split 50/50 by member. Incremental billed business:

$$
\Delta = \bar Y_{\text{treatment}} - \bar Y_{\text{control}}
$$

with a bootstrap 95% interval. Calibration is realized incremental \$ / predicted incremental \$. A ratio near $1$ means the offer model is on scale, not just ranking people.

---

## Layout

```
data/                      DGP + config (loads calibrated knobs)
data/calibrate_external.py
data/external/             committed snapshots (calibration.json, real_features.json)
notice/                    presence reconstruction + PPML gravity
explain/                   rule + PU cause classifier
match/                     submodular targeting + causal forest / X-learner
uplift/                    randomized holdout
dashboard/                 Streamlit app
notebooks/                 demo walkthrough
outputs/                   generated CSVs and metrics
run_pipeline.py
```

Dashboard colours from the NEMU pitch deck: charcoal `#1E1E1E`, white `#FFFFFF`, neon lime `#A7FC04`, Montserrat. Cause colours follow the wordmark (blue / lime / terracotta / purple).
# amex-26-nemu
