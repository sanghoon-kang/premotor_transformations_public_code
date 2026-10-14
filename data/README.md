# data

- `behavior.csv`: scanning-phase behavior, one row per completed trial. `manipulation_type` is `rotation` or `mirror` (reflection); `critical` means matched and `uncritical` unmatched; `which_target` is 1–4 (matched targets are 1 and 3); `initial_heading_deg` is the initial cursor direction relative to the target and `endpoint_angle_deg` the endpoint direction (degrees); `rt` and `mt` are in seconds.
- `tutorial.csv`: tutorial-phase behavior (n = 24), same columns, with `block` 1 = Rotation and 2 = Reflection.
- `rsm/<subject>_<ROI>.npy`: 8×8 similarity matrices (Spearman correlations between odd- and even-run condition patterns), conditions ordered rot1–4, ref1–4.
- `rsa_betas.csv`: RSA regression coefficients (Rot/Ref, Target, Move, Trial; fit with an intercept) per subject and ROI.
- `overlap.csv`: rule–movement overlap (`sq_overlap`) per subject and ROI.
