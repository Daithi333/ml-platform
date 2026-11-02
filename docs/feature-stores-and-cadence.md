# Feature Stores, Materialisation, and Retraining Cadence

## What a Feature Store Is

A feature store has two distinct parts:

**Registry** — stores feature *definitions*. Each definition names a feature, describes its
computation, and identifies the entity it attaches to (e.g. `user_id`). Written once,
versioned. This is the source of truth for how each feature is computed.

**Stores** — hold pre-computed *values*, materialised from raw data by applying definitions.
There are two stores, serving different consumers:

| Store | Technology (local) | Technology (prod) | Consumer |
|-------|-------------------|-------------------|----------|
| Offline | Parquet | Data warehouse / S3 | Model training |
| Online | Redis | Redis / DynamoDB | Live inference |

The critical design principle: **training and serving do not compute features independently.
They both read from the respective store.** A shared materialisation process computes the
values and writes them — one computation, two destinations.

```
Raw data (warehouse / event stream)
        |
        v
  Feature definition (e.g. txns_last_hour = COUNT(txns WHERE ts BETWEEN t-1h AND t))
        |
   [materialisation job]
        |
        +---> Offline store (Parquet): historical values, point-in-time correct
        |
        +---> Online store (Redis): latest value per entity, continuously updated

Training  --> reads pre-computed values from offline store (lookup, no computation)
Serving   --> reads pre-computed values from online store (lookup, no computation)
```

This eliminates training/serving skew — the subtlest and most damaging production ML bug,
where the model trains on one version of a feature and serves on a different version.

### Point-in-time correctness

When building a training dataset from history, each feature value must be computed as it
would have appeared *at the time of that historical event* — not using future data.

Example: training on a transaction from 2pm Tuesday. `txns_last_hour` must only count
transactions before 2pm Tuesday, not after. Using future data is **data leakage** — the
model appears to perform well in training but fails in production because it cheated with
information it won't have live.

Feature stores handle this with time-travel joins over the offline store.

### Some features require live arithmetic

A feature like `amount_vs_user_avg` depends on the current transaction's amount, which
only exists at request time. In practice:
- The user's historical average is pre-computed and stored (online store)
- The ratio is computed at request time: `incoming_amount / stored_avg`

The expensive, skew-prone part (the average) is still handled by the feature store.
Only trivial arithmetic using the live request happens outside it.

---

## Materialisation Cadence

**How often you materialise depends on how fast the underlying data changes and how much
staleness the prediction can tolerate.** This is decided per-feature, not globally.

| Feature | Data velocity | Cadence |
|---------|--------------|---------|
| `txns_last_5min` | Changes every second | Streaming (updated per event) |
| `user_avg_amount_90d` | Drifts slowly | Daily batch |
| `merchant_risk_score` | Weekly shifts | Daily or weekly batch |
| `account_age_days` | Increments daily | Daily batch |

The guiding question for each feature: **if this value is X hours old, does the prediction
still hold?**

A velocity feature like `txns_last_5min` is worthless if stale — the whole point is
detecting rapid-fire activity. But `account_age_days` being a few hours old changes
nothing.

Rules of thumb:
- Fast-changing, decision-critical → streaming (real-time, per-event update to online store)
- Slow-changing → batch (nightly is usually sufficient)
- Offline store (training data) → almost always batch, rebuilt ahead of each training run

Streaming is more expensive and operationally complex. Only use it where staleness
measurably degrades prediction quality.

---

## Model Retraining Cadence

Materialisation keeps feature values fresh. Retraining updates the model's understanding
of *what patterns mean*. These are independent concerns driven by different forces.

You retrain when learned relationships go stale. Three triggers:

### 1. Scheduled (time-based)

Retrain on a fixed schedule regardless of observed drift. Simple, predictable, low
operational overhead. Good when patterns drift at a known rate.

Examples from this platform:
- Engagement predictor → daily (behaviour shifts with daily content cycle)
- Demand forecaster → weekly (inventory patterns are stable week-to-week)
- Content classifier → weekly (topic patterns drift slowly)

Start here. Add smarter triggers once monitoring exists.

### 2. Drift-triggered

Monitor input feature distributions and prediction distributions. When they shift beyond
a threshold, trigger retraining. More responsive than a fixed schedule — you retrain when
you need to, not on an arbitrary clock.

Requires monitoring infrastructure (Prometheus, drift detection) before it's viable. This
is why monitoring is phase 5 in the implementation order and retraining automation is
phase 10 — you cannot trigger on signals you aren't measuring.

### 3. Performance-triggered

Watch live accuracy, precision, or recall. When it drops below a threshold, retrain.

The catch for fraud: the true label (chargeback = confirmed fraud) may arrive 30-60 days
after the transaction. Performance signals are delayed, so this trigger lags reality. It
is a useful backstop but not a primary signal for high-velocity models.

---

## The Interaction Between Materialisation and Retraining

This is where the two cadences must be coordinated.

**Rule: when you retrain, the training set must use the same feature definitions
currently serving production.** Otherwise the model trains on a different feature
space than it serves on — skew, again.

If a feature definition changes (e.g. `txns_last_hour` now excludes refunds), you must:
1. Update the definition in the registry
2. Re-materialise the offline store using the new definition
3. Then retrain

In the reverse direction: **a model version is bound to the feature versions it trained
with.** If you run model v3 (trained on feature set A) alongside a candidate model v4
(trained on feature set B), the serving layer must supply the correct feature version to
each. This is where model versioning, feature versioning, and deployment strategies
(canary, shadow mode) converge.

---

## Decision Framework Summary

**Materialisation cadence:** Per feature. Ask "what is the cost of this value being stale?"
Match cadence to velocity. Use streaming only where it earns its cost.

**Retraining cadence:** Per model. Ask "how fast does the world change relative to what the
model learned?" Start scheduled, add drift triggers once monitoring exists.

**Coordination rule:** Feature definition changes → re-materialise offline store → retrain.
Never retrain on stale features.

---

## Mapping to Platform Phases

| Phase | Component | Why it must come first |
|-------|-----------|----------------------|
| 2 | Feature store | Establishes consistent feature computation |
| 3 | Model registry + promotion | Versions models, binds them to feature versions |
| 5 | Monitoring | Measures drift — prerequisite for triggered retraining |
| 10 | Retraining automation | Requires monitoring signals to trigger on |

You cannot do smart retraining until monitoring tells you when patterns drifted.
Scheduled retraining is the pragmatic default until that infrastructure exists.
