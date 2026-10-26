# Retention simulator — formulas, assumptions, limits

The simulator (`src/churnguard/retention_simulator.py`, page *Retention Simulator*) compares retention-campaign
strategies **under assumptions you enter**. It is a scenario calculator. It is **not** a forecast, a measurement of
benefit, or evidence that any campaign works.

## What the model does and does not provide

* It provides, for each customer, a **probability of churn** `p_i` (checked for calibration on validation data).
* It does **not** provide the effect of a campaign. Churn data alone cannot tell who would have stayed anyway,
  who responds to an offer, or whether an offer triggers other behaviour. That effect is the assumption **Q** below.
* Not every at-risk customer is recoverable; Q encodes that, and defaults to a deliberately modest value.

## Notation and units

| Symbol | Meaning | Unit |
|---|---|---|
| `C` | cost of contacting one customer | currency per customer |
| `V` | value of one retained customer (e.g. margin over the horizon you care about) | same currency |
| `Q` | assumed campaign success rate: share of would-be churners who are retained, in [0, 1] | none |
| `S` | set of contacted customers, `N = ⎮S⎮` | customers |
| `p_i` | model probability that customer `i` churns | none |

Currency is whatever you use for `C` and `V`; the app never converts or labels it.

## Formulas

```
cost                   = N × C
R  (expected churners) = Σ_{i ∈ S} p_i
churn avoided          = Q × R
gross benefit          = V × Q × R
net benefit            = V × R × Q − N × C
ROI                    = net benefit / cost            (undefined when cost = 0)
break-even probability p* = C / (V × Q)                (a contact has positive expected value iff p_i > p*)
break-even success rate Q* = N × C / (V × R)           (the Q at which a strategy's net benefit is zero)
```

`R` is only "expected churners" if the probabilities are calibrated; the app warns when the model's mean predicted
probability differs from the observed rate by more than 3 points on validation data. On the held-out test set of
the committed model the probabilities sum to 376.5 for 1 409 customers while 374 churned historically — a
consistency check, not a guarantee.

## Strategies compared

| # | Strategy | Customers contacted |
|---|---|---|
| 1 | No campaign | none (reference: cost 0, benefit 0) |
| 2 | Contact everyone | all |
| 3 | Top *k* by risk | the *k* highest `p_i` |
| 4 | Risk ≥ threshold | every customer with `p_i ≥ t` |
| 5 | Budget-limited | the highest `p_i` until the budget (`⌊budget / C⌋` contacts) or the contact limit is reached |

Only strategy 5 is shaped by the budget and the contact limit. The others are shown unconstrained and flagged
**"within budget & contact limit? NO"** when they would exceed a limit, so ignoring a constraint stays visible.
Ties are broken by input order. `max_contacts` and `budget` are optional; with `C = 0` the budget cannot bind.

An optional extra column, *churners among contacted (historical)*, counts how many contacted customers actually
churned in the data. It is a reference for calibration, **not** an effect of the campaign.

## Worked example (illustrative defaults)

Held-out test customers of the committed model (1 409 customers), with `C = 15`, `V = 250`, `Q = 0.20`, budget 3 000,
max 500 contacts, threshold 0.50, k = 300. These inputs are **placeholders, not estimates from data**:

| Strategy | Contacted | Within limits | Cost | Expected churners targeted (R) | Gross | Net | ROI |
|---|---|---|---|---|---|---|---|
| No campaign | 0 | yes | 0 | 0 | 0 | 0 | n/a |
| Contact everyone | 1 409 | no | 21 135 | 376.5 | 18 827 | −2 308 | −11 % |
| Top 300 by risk | 300 | no | 4 500 | 198.3 | 9 915 | 5 415 | 120 % |
| Risk ≥ 0.50 | 304 | no | 4 560 | 200.3 | 10 015 | 5 455 | 120 % |
| Budget-limited | 200 | yes | 3 000 | 143.3 | 7 163 | 4 163 | 139 % |

Break-even probability here is `p* = 15 / (250 × 0.20) = 0.30`. The ranking depends strongly on `Q`: with the same
other inputs, `Q = 0.05` makes every campaign lose money (net −1 209 for the budget-limited one), `Q = 0.30` makes even
contacting everyone profitable (net +7 105). Because no input here is justified by data, **no strategy is recommended**:
the page shows the sensitivity of net benefit to `Q` for exactly that reason, and declares no winner.

## Simplifications and limits

* A single `V` for every customer. Real value varies (a customer paying 100/month is not worth the same as one paying 20);
  the module has no per-customer value yet.
* A single `Q` for every customer and every strategy. In reality response differs by segment, and the customers a
  model ranks highest are not necessarily the most persuadable.
* No fixed campaign cost, no discount offered to retained customers, no time horizon or discounting, no
  cannibalisation or contact fatigue, no cost of contacting customers who would not have left (beyond `C`).
* Each contacted customer is counted independently, and a single campaign is modelled: no budget carry-over, no
  repeated contacts.
* Probabilities come from a model trained on a sample data set; they transfer to other customers only after your own
  validation.

Use the simulator to *explore* "what would I have to believe for this campaign to pay off?" (`p*`, `Q*`, sensitivity),
not to claim a result.
