# Research demonstration methodology

This independent demonstration generates synthetic story-feature vectors and Fibonacci-style story-point targets. It is not the original NUST thesis dataset or a validated industrial model.

## Predictive pipeline
- Holdout testing set is 20% of observations.
- Development data is partitioned into a training set and untouched calibration set.
- Random forest regression fits on training rows only.
- Absolute calibration residuals determine the conformal cutoff at finite-sample quantile.
- Prediction interval is [prediction - cutoff, prediction + cutoff], truncated to the simulated domain [1, 13].
- Holdout empirical coverage, MAE and average interval width are evaluated independently.

**Interpretation:** Split conformal prediction targets marginal coverage only under exchangeability. Distribution shift, label design, time dependence, selection bias, organization/team differences, and truncated intervals can alter achieved coverage. Production applicability and original research novelty require separate studies, experiments and proper citations.
