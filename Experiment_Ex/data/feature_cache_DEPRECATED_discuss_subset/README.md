# DEPRECATED — Stale 259-sample Discuss-subset feature caches

These eight `.npz` files were produced by an earlier debugging run that
fed the SCB5_Discuss subset (a single-class, 259-image set, intentionally
excluded from all quantitative analyses; see Section 3.1 of the paper)
through the encoders but mis-named the outputs as if they were the
TeacherBehavior / HandriseReadWrite / BowTurnHead validation caches.

**Do not use these files for any evaluation.** Each file contains
259 samples and degenerate labels (all 0 for single-label datasets;
multi-hot vectors over 259 Discuss images for the multi-label case).

The canonical, full-coverage feature caches live in
`<repo>/data/feature_cache/` and contain:

  - `*_teacher_behavior_validation.npz`     (3240 samples, 8 multi-label)
  - `*_handrise_readwrite_validation.npz`   (1671 samples, 3-way single-label)
  - `*_bow_turnhead_validation.npz`         ( 505 samples, 2-way single-label)

This directory is retained only to preserve provenance of the deprecated run;
it can be removed without affecting any reported result.
