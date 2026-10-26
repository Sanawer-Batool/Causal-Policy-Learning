# AI log

- 2026-10-26: The user corrected an earlier suggestion to rescale half-sample
  intervals by `sqrt(1/2)`. Sampling `n/2` rows without replacement has the
  finite-population correction `(1/b - 1/n) sigma^2 = sigma^2/n`, so the raw
  half-sample percentile interval is already on the full-sample variance
  scale. No rescaling should be added.
- 2026-10-26: Provisional Phase 4 decision: use the `subsample` scheme for the
  full development audit because it avoids duplicate-row leakage and was at
  least as conservative in the seed-0, kappa-0 smoke test. The cost rationale
  was withdrawn: subsample was faster at kappa=0 (199 s vs 339 s) but slower at
  kappa=1 (307 s vs 285 s), and runtimes may have been contaminated by overlap
  with the development run. This remains a one-seed, one-setting decision until
  variance calibration and confirmation; divergence requires revisiting it.
- 2026-10-26: Keep S, DR-S, and Forest as headline methods, with X as a
  reference. Leave LightGBM threading unchanged because the n_jobs=4 smoke
  runtimes scaled sensibly without evidence of oversubscription.
- 2026-10-26: Wider subsample intervals do not by themselves prove that
  duplicate-row leakage caused bootstrap under-dispersion. Half-sample fits
  use n/2 rows with hyperparameters tuned for n, so they are a differently
  regularized estimator; the smooth-statistic scaling argument is insufficient.
- 2026-10-26: Refined F3: variance-only intervals fail whenever bias is at least
  the half-width. Split F4/F7: F7a is shrinkage bias present under random
  assignment, while F7b is regularization-induced confounding present at kappa0
  and absent at rho0. F8 records that DR-S and X intervals are about 4 wide,
  making their coverage real but uninformative. F9 records
  estimator-independent confounding bias with the same arm ordering across
  learners; an earlier F4/F7 merge was premature.
- 2026-10-26: Variance calibration at kappa=0 using 30 independent datasets
  selected subsample for S overall (ratios to independent-dataset SD:
  bootstrap [1.297, 1.368, 1.028], subsample [1.118, 1.227, 1.158])
  and bootstrap for DR-S (bootstrap [1.109, 1.240, 1.094], subsample
  [1.416, 1.502, 1.422]). These results do not isolate duplicate-row leakage;
  they reflect the full estimator behavior. X remains on subsample pending
  its own calibration, and Forest uses native intervals.
