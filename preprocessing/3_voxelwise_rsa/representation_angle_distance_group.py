#!/usr/bin/env python3
"""
subspace_angle_group.py

Group stats and figure for the rule-vs-movement geometry. Interactive.

WHAT IS TESTED
The subject script gives, per subject per ROI, two cross-fold cosines between
the rule axis and the target-controlled movement axis. Their product,
sq_overlap, is an unbiased estimate of the squared cosine — unbiased because
the two estimates come from disjoint halves of the data.

The test is a one-sample t-test of sq_overlap against zero, one-sided, on two
a priori ROIs (left PMd, left M1) with no correction across them. Same test
family as the RSA betas elsewhere in the manuscript, minus the correction,
because two pre-specified regions do not need one.

WHY THE TWO SIDES ARE ORTHOGONAL BY DESIGN
Within each rule the four required movements form antipodal pairs (110/290 and
150/330 under rotation, 110/290 and 70/250 under mirror), so each rule's
movements sum to the zero vector. The rot-minus-ref contrast carries no net
movement, and neither movement weight can absorb the rule difference. Any
overlap measured is in the data, not the design.

READ THE RELIABILITY COLUMNS FIRST. If rel_rule / rel_cos / rel_sin are near
zero, the vectors do not replicate across folds and their overlap is not
interpretable.
"""

# %% imports
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from utils.pipeline_config import cfg, ROI_SETS, roi_strings

mpl.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "pdf.fonttype": 42})
MM = 1 / 25.4

# %% config
# PRIMARY measure: sq_overlap = cos1 * cos2, the product of the two cross-fold
# cosines. They use disjoint halves of the data, so they are independent
# estimates of the same cosine mu, and their product is unbiased for mu^2.
#
# Squared, because the sign of a cosine between two difference-vectors depends
# on which way each contrast was written AND on which way a given subject's
# rule axis leans — and there is no reason subjects should agree on that. A
# signed average can be zero while every subject has real overlap.
#
# Unbiased without any correction, because the two estimates are independent:
# a naive |cos| or cos^2 from a single estimate would be inflated by noise.
#
# One-sided is legitimate — a squared overlap cannot be negative, so the
# constraint is real rather than a guess about which sign to expect.
MEASURE = "sq_overlap"
ALTERNATIVE = "greater"

# Which ROIs to test. The two a priori regions come from the MDS and
# brain-behaviour results; with two pre-specified regions no correction across
# them is needed. Set to None to run every ROI as an exploratory table.
ROI_SUBSET = ["PMd_left_volume", "M1_left_volume"]
BAR_COLOR = "#A6CEE3"      # Rot/Ref blue, as in the manuscript


def format_p(p):
    """
    Figure p-value convention, shared across every figure script — keep the
    copies identical. Two decimals, three below .01 so a small p is not
    rounded up to a threshold (0.007 would otherwise print as 0.01), and a
    floor at .001.
    """
    if p < 0.001:
        return "p < 0.001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


# %% load
path = os.path.join(cfg.rsa_dir, f"subspace_angle_{cfg.event}_{cfg.censor_label}.csv")
df = pd.read_csv(path)
print(f"{len(df)} rows | {df.subject.nunique()} subjects | {df.roi.nunique()} ROIs")

roi_order = [r for r in roi_strings(list(ROI_SETS)) if r in set(df.roi)]
# Labels for EVERY ROI present, not just the tested subset — the exploratory
# correlation table below needs names for the ROIs that ROI_SUBSET excludes.
label = {r: ("L " if "left" in r else "R ") + r.split("_")[0]
         for r in roi_order}
if ROI_SUBSET is not None:
    roi_order = [r for r in ROI_SUBSET if r in roi_order]
    missing = [r for r in ROI_SUBSET if r not in roi_order]
    if missing:
        print(f"[WARN] requested ROIs absent from the data: {missing}")
print(f"testing {len(roi_order)} ROI(s): {[r.split('_')[0] for r in roi_order]}")

print(f"{MEASURE} defined: {df[MEASURE].notna().sum()} / {len(df)} "
      f"({100 * df[MEASURE].notna().mean():.0f}%)")

# %% group stats, per ROI
rows = []
for roi in roi_order:
    d = df[df.roi == roi].dropna(subset=[MEASURE])
    if len(d) < 4:
        print(f"[SKIP] {label[roi]}: {len(d)} usable subjects")
        continue
    t, p = stats.ttest_1samp(d[MEASURE], 0.0, alternative=ALTERNATIVE)
    m = d[MEASURE].mean()
    rows.append(dict(
        roi=roi, label=label[roi], family=d.family.iloc[0], n=len(d),
        mean_sq_overlap=m, sem_sq_overlap=stats.sem(d[MEASURE]),
        # effect size on the cosine scale: sqrt of the mean squared overlap
        overlap=np.sqrt(max(m, 0.0)),
        t=t, p_uncorrected=p,
        mean_signed=d.cos_mean.mean(),   # do subjects agree on direction?
        t_signed=stats.ttest_1samp(d.cos_mean, 0.0).statistic,
        rel_rule=d.rel_rule.mean(), rel_move=d.rel_move.mean(),
    ))

summ = pd.DataFrame(rows)
summ["p_fdr"] = (summ.groupby("family")["p_uncorrected"]
                 .transform(lambda g: multipletests(g, method="fdr_bh")[1]))

# ---- PRIMARY TEST: the dissociation ----------------------------------------
# Two significant-vs-not one-sample tests do NOT establish that the ROIs
# differ — that is the comparison-of-significance error. The paired test asks
# the question directly, and is more powerful because each subject contributes
# both values, so shared subject-level noise cancels in the difference.
PAIRED = None
if len(roi_order) == 2:
    a, b = roi_order
    wide = (df[df.roi.isin(roi_order)]
            .pivot(index="subject", columns="roi", values=MEASURE).dropna())
    d = wide[a] - wide[b]
    t_p, p_p = stats.ttest_rel(wide[a], wide[b], alternative="greater")
    PAIRED = dict(a=label[a], b=label[b], n=len(wide),
                  mean_diff=d.mean(), sem_diff=stats.sem(d),
                  t=t_p, p=p_p, cohens_dz=d.mean() / d.std(ddof=1))
    print(f"\nPAIRED (primary): {PAIRED['a']} − {PAIRED['b']}   "
          f"mean diff = {PAIRED['mean_diff']:.4f} "
          f"(SEM {PAIRED['sem_diff']:.4f}), "
          f"t({PAIRED['n']-1}) = {PAIRED['t']:.2f}, p = {PAIRED['p']:.4f}, "
          f"dz = {PAIRED['cohens_dz']:.2f}")

out_csv = os.path.join(cfg.rsa_dir, f"subspace_angle_group_{cfg.event}.csv")
summ.to_csv(out_csv, index=False)
print(f"\nSaved {out_csv}")

pd.set_option("display.width", 220)
print("\nPer-ROI decomposition (which side drives the paired result)")
print(summ[["label", "n", "overlap", "mean_sq_overlap", "sem_sq_overlap",
            "t", "p_uncorrected", "p_fdr", "mean_signed", "t_signed",
            "rel_rule", "rel_move"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n'overlap' is on the cosine scale: sqrt of the noise-corrected mean "
      "squared overlap.\nThe test is on mean_sq_overlap vs 0, one-sided.\n"
      "mean_signed / t_signed are secondary: they ask whether subjects happen "
      "to agree on\nthe DIRECTION of the overlap, which the primary measure "
      "does not require.")

# %% figure
# Same shape as the Figure 2 swarm panels: subject dots, mean +/- SEM, stars
# from the FDR-corrected p. Plotting a signed component means the dots and the
# group mean ARE the same quantity, so the overlay is honest — unlike a
# magnitude, where |mean of vectors| and |vector| differ by construction.
c = MEASURE
fig, ax = plt.subplots(figsize=(max(78, 42 * len(summ)) * MM, 74 * MM))
x = np.arange(len(summ))
rng = np.random.default_rng(0)

for xi, roi in zip(x, summ["roi"]):
    v = df.loc[(df.roi == roi) & df[c].notna(), c].to_numpy()
    ax.scatter(xi + rng.normal(0, 0.06, len(v)), v, s=14, color=BAR_COLOR,
               edgecolor="black", linewidths=0.3, alpha=0.7, zorder=2)

ax.errorbar(x, summ["mean_sq_overlap"], yerr=summ["sem_sq_overlap"], fmt="o",
            color="black", markersize=3.5, capsize=2.5, linewidth=1.0, zorder=4)
ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

# Per-ROI statistics on the figure: one one-sample t-test against zero,
# one-sided. With two pre-specified regions these are reported UNCORRECTED —
# BH across two tests barely penalises, and quoting a corrected value would
# overstate what the correction did.
# All annotations on ONE common line above the highest data point, so they
# cannot collide with each other or drift into the title.
_lo, _hi = df[c].min(), df[c].max()
_span = _hi - _lo
_ann_y = _hi + 0.10 * _span
for xi, (_, row) in zip(x, summ.iterrows()):
    pval = row["p_uncorrected"]
    ptxt = format_p(pval)
    ax.text(xi, _ann_y, f"t({row['n'] - 1:.0f}) = {row['t']:.2f}\n{ptxt}",
            ha="center", va="bottom", fontsize=8, linespacing=1.4,
            color="black" if pval < .05 else "0.45")
ax.set_ylim(_lo - 0.10 * _span, _ann_y + 0.34 * _span)
ax.set_xlim(-0.7, len(summ) - 0.3)

# bracket showing the primary paired comparison
if PAIRED is not None:
    _y = _ann_y + 0.20 * _span
    ax.plot([0, 0, 1, 1], [_y, _y + 0.04*_span, _y + 0.04*_span, _y],
            color="black", lw=1.0, clip_on=False)
    _pt = format_p(PAIRED["p"])
    ax.text(0.5, _y + 0.06*_span,
            f"paired: t({PAIRED['n']-1}) = {PAIRED['t']:.2f}, {_pt}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            clip_on=False)

ax.set_xticks(x)
ax.set_xticklabels(summ["label"], fontsize=9, rotation=0)
ax.set_ylabel("Squared overlap\n(rule vs movement)", fontsize=9)
ax.set_title("Rule vs movement, target controlled", fontsize=10, pad=8)
sns.despine(ax=ax, trim=True)
plt.tight_layout()

for ext in ("pdf", "png"):
    p_out = os.path.join(cfg.fig_dir, f"fig_subspace_overlap_{cfg.event}.{ext}")
    fig.savefig(p_out, dpi=300, bbox_inches="tight")
    print(f"saved {p_out}")
plt.show()


# %% ── caching correlation ────────────────────────────────────────────────
# Does a subject's rule-movement overlap track their behavioural caching?
#
# DIRECTIONAL PREDICTION, stated before running: if a subject caches, their
# "rule" representation IS a set of movements, so rule and movement should be
# more entangled. Predicted POSITIVE correlation between sq_overlap and the
# caching index. One-sided is legitimate because sq_overlap is sign-free — the
# prediction is about the direction of a relationship, not about which way a
# contrast happens to be written.
#
# THE CONFOUND. sq_overlap is attenuated by each subject's data quality, so it
# partly indexes SNR rather than geometry. If data quality relates to behaviour
# at all — and subjects who move less have both cleaner data and often
# different behaviour — a correlation appears with no true geometry behind it.
# In simulation, a data-quality/behaviour correlation of 0.5 produces a
# spurious r of about 0.38, which is the size of result one would celebrate.
#
# So two things are reported for every ROI:
# r_overlap_vs_reliability is reported as a diagnostic: if sq_overlap does not
# track reliability across subjects, the confound is not active and r_raw can
# be read at face value.
#
# NOTE ON SHAPE. The two readouts in caching_correlation.py pair a within-
# subject NEURAL difference (uncritical - critical) with a within-subject
# BEHAVIOURAL difference. sq_overlap is a single value per subject, so this
# pairs a level with a difference — a different question, not a third version
# of the same one.
#
# caching_index() is duplicated from caching_correlation.py rather than
# imported: that file has no __main__ guard, so importing it runs its driver.

from scipy.stats import pearsonr, spearmanr

CACHE_MEASURES = ["rt", "rt_raw", "var_initial", "var_endpoint"]
# TWO-TAILED for the correlations. The directional prediction above was formed
# while developing this analysis rather than beforehand, which is weaker ground
# for a one-sided test than the rule-beta/error correlation, where the
# direction was predicted in advance. The two-tailed value is simply twice the
# one-tailed one here (the t null is symmetric), so nothing is hidden by using
# it — it is just the conservative choice for an exploratory result.
# All correlation p-values below are TWO-TAILED. CORR_DIRECTION only says which
# direction the one-tailed value is computed for, reported alongside.
CORR_DIRECTION = "greater"
A_PRIORI = ["PMd_left_volume", "M1_left_volume"]   # from the MDS / behaviour work
# Other sensorimotor regions, printed for reference only — no correction, no
# figure. The a priori pair above is what the claim rests on; these say whether
# the pattern is specific to left PMd or shared across the motor network.
# Set REFERENCE_REGIONS to None to print every remaining ROI instead.
REFERENCE_REGIONS = ["PMd", "PMv", "M1", "S1", "SPL"]

_avail = [r for r in roi_strings(list(ROI_SETS))
          if r in set(df.roi) and r not in A_PRIORI]
EXPLORATORY = ([r for r in _avail if r.split("_")[0] in REFERENCE_REGIONS]
               if REFERENCE_REGIONS is not None else _avail)
ERR_COL = {"initial": "initial_heading_deg", "endpoint": "endpoint_angle_deg"}

_B = pd.read_csv(cfg.behav_csv)
EXPLORATORY_TABLES = []


def caching_index(measure="rt"):
    """
    Unmatched - Matched, as in caching_correlation.py.

    "rt" / "mt" / "var_*" take |mirror - rotation| within each trial type first,
    so they index the size of the RULE difference and then contrast it across
    matched and unmatched trials.

    "rt_raw" / "mt_raw" skip that step: they are simply mean RT on unmatched
    minus mean RT on matched, collapsed across rule. A simpler quantity — it
    asks whether unmatched trials are slower, not whether the rules differ
    more there.
    """
    if measure.endswith("_raw"):
        col = measure.replace("_raw", "")
        g = _B.groupby(["subject", "critical"])[col].mean().reset_index()
        w = g.pivot(index="subject", columns="critical", values=col)
        return (w["uncritical"] - w["critical"]).rename("caching_index")
    if measure in ("rt", "mt"):
        col, fn = measure, "mean"
    elif measure in ("var_initial", "var_endpoint"):
        col, fn = ERR_COL[measure.split("_")[1]], "var"
    else:
        raise ValueError(f"unknown caching measure: {measure}")
    g = (_B.groupby(["subject", "critical", "manipulation_type"])[col]
           .agg(fn).reset_index())
    w = g.pivot_table(index=["subject", "critical"],
                      columns="manipulation_type", values=col)
    w["delta"] = (w["mirror"] - w["rotation"]).abs()
    ci = w["delta"].reset_index().pivot(index="subject", columns="critical",
                                        values="delta")
    return (ci["uncritical"] - ci["critical"]).rename("caching_index")


def correlate_overlap(caching, rois=A_PRIORI):
    rows = []
    for roi in rois:
        d = df[df.roi == roi].set_index("subject")
        m = pd.concat([caching, d[[MEASURE, "rel_rule", "rel_move"]]],
                      axis=1, join="inner").dropna()
        if len(m) < 6:
            continue
        x, y = m[MEASURE].to_numpy(), m["caching_index"].to_numpy()
        rel = (m["rel_rule"] * m["rel_move"]).to_numpy()

        r_raw, p_raw = pearsonr(x, y)                    # two-tailed
        r_sp, p_sp = spearmanr(x, y)
        p_raw_1t = p_raw/2 if ((r_raw > 0) == (CORR_DIRECTION == "greater")) \
            else 1 - p_raw/2
        # Diagnostic only: if overlap tracks data quality, the correlation may
        # be SNR rather than geometry. Near zero means the confound is inactive.
        r_snr, p_snr = pearsonr(x, rel)

        rows.append(dict(roi=roi, label=label[roi], n=len(m),
                         r_raw=r_raw, p_raw=p_raw, p_raw_1tailed=p_raw_1t,
                         r_spear=r_sp, p_spear=p_sp,
                         r_overlap_vs_reliability=r_snr, p_rel=p_snr))
    return pd.DataFrame(rows)


def scatter_overlap(caching, tab, measure_name, out_name):
    n = len(tab)
    fig, axes = plt.subplots(1, n, figsize=(62 * n * MM, 58 * MM), squeeze=False)
    for k, (_, r) in enumerate(tab.iterrows()):
        ax = axes[0, k]
        d = df[df.roi == r["roi"]].set_index("subject")
        m = pd.concat([caching, d[[MEASURE]]], axis=1, join="inner").dropna()
        ax.scatter(m["caching_index"], m[MEASURE], s=18, color=BAR_COLOR,
                   edgecolor="black", linewidths=0.3)
        b1, b0 = np.polyfit(m["caching_index"], m[MEASURE], 1)
        xs = np.linspace(m["caching_index"].min(), m["caching_index"].max(), 50)
        ax.plot(xs, b0 + b1*xs, color="0.3", lw=1)
        ax.axhline(0, color="0.85", lw=0.6)
        ax.set_title(f"{r['label']}  r = {r['r_raw']:.2f}  (p = {r['p_raw']:.3f})",
                     fontsize=8)
        ax.set_xlabel(f"Caching index [{measure_name}]", fontsize=7)
        ax.set_ylabel("squared overlap", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        pth = os.path.join(cfg.fig_dir, f"{out_name}.{ext}")
        fig.savefig(pth, dpi=300, bbox_inches="tight")
        print(f"saved {pth}")
    plt.show()


for meas in CACHE_MEASURES:
    ci = caching_index(meas)

    tab = correlate_overlap(ci, rois=A_PRIORI)
    print(f"\n===== caching index [{meas}] vs {MEASURE} "
          f"(p two-tailed; p_raw_1tailed is the one-sided value) =====")
    if tab.empty:
        print("  no ROIs with enough overlapping subjects")
        continue
    print("A PRIORI")
    print(tab.round(3).to_string(index=False))
    print("  r_overlap_vs_reliability near 0 -> the SNR confound is not active.")

    # Reference only. Uncorrected and not plotted; with ~20 ROIs and three
    # caching measures some of these will reach p < .05 by chance alone.
    ref = correlate_overlap(ci, rois=EXPLORATORY)
    if not ref.empty:
        print(f"\nOTHER SENSORIMOTOR REGIONS (reference only, uncorrected, "
              f"{len(ref)} ROIs x {len(CACHE_MEASURES)} measures)")
        print(ref[["label", "n", "r_raw", "p_raw", "r_spear", "p_spear"]]
              .sort_values("p_raw").round(3).to_string(index=False))
        ref.insert(0, "measure", meas)
        EXPLORATORY_TABLES.append(ref)

    scatter_overlap(ci, tab, meas, f"fig_overlap_vs_caching_{meas}_{cfg.event}")


# %% exploratory correlations, saved for reference
if EXPLORATORY_TABLES:
    _ref = pd.concat(EXPLORATORY_TABLES, ignore_index=True)
    _p = os.path.join(cfg.rsa_dir,
                      f"overlap_caching_exploratory_{cfg.event}.csv")
    _ref.to_csv(_p, index=False)
    print(f"\nSaved exploratory correlations: {_p}")
    print(f"  {len(_ref)} ROI x measure combinations; "
          f"{int((_ref.p_raw < .05).sum())} at p < .05 uncorrected "
          f"(expected by chance: {0.05 * len(_ref):.1f})")


# %% control analyses
# Three things that could produce the overlap-caching correlation without any
# geometry behind it:
#
#   (1) VECTOR STRENGTH. If caching tracks how strong the rule or movement
#       vector is, rather than how much they overlap, that shows up as a
#       correlation with mag_rule / mag_move.
#
#   (2) RELIABILITY. sq_overlap is attenuated by data quality, so it partly
#       indexes SNR. If caching correlates with rel_rule / rel_move, the
#       overlap correlation may be measurement quality. Reported alongside a
#       partial correlation holding reliability constant.
#
#   (3) A SIMPLER BEHAVIOURAL INDEX. rt_raw is unmatched-minus-matched RT with
#       no |mirror - rotation| step, so it does not presuppose the rule-
#       difference framing. Included in CACHE_MEASURES above.
#
# A result that survives all three is about overlap. One that vanishes under
# any of them was that thing instead.

def _partial(x, y, Z):
    """corr(x, y) controlling for the columns of Z, with a one-sided p."""
    Z = np.column_stack([np.ones(len(x))] + [np.asarray(z) for z in Z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r, _ = pearsonr(rx, ry)
    dof = len(x) - Z.shape[1] - 1
    t = r * np.sqrt(dof / max(1e-12, 1 - r**2))
    return r, 2 * stats.t.sf(abs(t), dof)          # two-tailed


CONTROL_COLS = ["sq_overlap", "mag_rule", "mag_move", "rel_rule", "rel_move"]
CONTROL_TABLES = []

# Are matched trials simply faster? Sam's prediction: more practice on matched
# targets should lower overall RT there, independent of any rule difference.
# rt_raw is exactly that contrast (unmatched - matched, collapsed across rule),
# so a paired t-test of it against zero is the test.
print(f"\n{'='*62}\nOVERALL RT: matched vs unmatched (collapsed across rule)"
      f"\n{'='*62}")
for _m in ("rt", "mt"):
    _raw = f"{_m}_raw"
    try:
        _d = caching_index(_raw).dropna()
    except (KeyError, ValueError):
        continue
    _t, _p = stats.ttest_1samp(_d, 0.0)
    print(f"  {_m}: unmatched - matched = {_d.mean():+.4f} s "
          f"(SEM {stats.sem(_d):.4f}), t({len(_d)-1}) = {_t:.2f}, "
          f"{format_p(_p)}")
    print(f"     -> unmatched slower in {int((_d > 0).sum())}/{len(_d)} "
          f"participants")
print("  positive = unmatched slower, i.e. matched trials faster overall")


for meas in CACHE_MEASURES:
    ci = caching_index(meas)
    print(f"\n===== CONTROLS: caching [{meas}] =====")
    for roi in A_PRIORI:
        d = df[df.roi == roi].set_index("subject")
        m = pd.concat([ci, d[CONTROL_COLS]], axis=1, join="inner").dropna()
        if len(m) < 6:
            continue
        y = m["caching_index"].to_numpy()

        rows = []
        for col in CONTROL_COLS:
            r, p = pearsonr(m[col], y)          # two-tailed
            rows.append(dict(predictor=col, r=r, p=p))
        tab = pd.DataFrame(rows)

        # overlap with the nuisance variables held constant
        r_mag, p_mag = _partial(m["sq_overlap"].to_numpy(), y,
                                [m["mag_rule"], m["mag_move"]])
        r_rel, p_rel_ = _partial(m["sq_overlap"].to_numpy(), y,
                                 [m["rel_rule"], m["rel_move"]])
        tab = pd.concat([tab, pd.DataFrame([
            dict(predictor="sq_overlap | magnitudes", r=r_mag, p=p_mag),
            dict(predictor="sq_overlap | reliability", r=r_rel, p=p_rel_),
        ])], ignore_index=True)

        tab.insert(0, "measure", meas)
        tab.insert(1, "roi", label[roi])
        tab["n"] = len(m)
        CONTROL_TABLES.append(tab)

        print(f"\n  {label[roi]}  (n = {len(m)})")
        print(tab[["predictor", "r", "p"]].round(3).to_string(index=False))

        raw = tab.loc[tab.predictor == "sq_overlap", "r"].iloc[0]
        rel_r = tab.loc[tab.predictor.isin(["rel_rule", "rel_move"]), "r"].abs().max()
        print(f"    |r| overlap = {abs(raw):.3f}   vs   "
              f"largest |r| for reliability = {rel_r:.3f}"
              f"   -> {'overlap is larger' if abs(raw) > rel_r else 'RELIABILITY IS LARGER'}")


# %% control figure
# Two questions, two panels:
#   A  for the headline caching index, does overlap stand out from the control
#      predictors? (is it geometry, or magnitude / data quality?)
#   B  for overlap alone, is the correlation specific to one caching index?
# Error bars are 95% CIs from the Fisher z transform — at n = 27 they are wide,
# and showing them is more honest than a bare point estimate.
CTRL = pd.concat(CONTROL_TABLES, ignore_index=True)
CTRL.to_csv(os.path.join(cfg.rsa_dir, f"overlap_controls_{cfg.event}.csv"),
            index=False)

def _ci(r, n):
    z = np.arctanh(np.clip(r, -0.999, 0.999))
    se = 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)

HEADLINE = "rt"
PRED_ORDER = ["sq_overlap", "sq_overlap | reliability", "sq_overlap | magnitudes",
              "rel_rule", "rel_move", "mag_rule", "mag_move"]
PRED_LABEL = {"sq_overlap": "overlap", "sq_overlap | reliability": "overlap | reliability",
              "sq_overlap | magnitudes": "overlap | magnitude",
              "rel_rule": "reliability (rule)", "rel_move": "reliability (move)",
              "mag_rule": "magnitude (rule)", "mag_move": "magnitude (move)"}
ROI_COLOR = {label[A_PRIORI[0]]: BAR_COLOR, label[A_PRIORI[1]]: "0.75"}

fig, axes = plt.subplots(1, 2, figsize=(180*MM, 72*MM),
                         gridspec_kw=dict(width_ratios=[1.45, 1], wspace=0.50))

# ---- A: predictors, headline measure ----
ax = axes[0]
sub = CTRL[CTRL.measure == HEADLINE]
ys = np.arange(len(PRED_ORDER))[::-1]
for k, (roi_lab, off) in enumerate(zip(ROI_COLOR, (+0.16, -0.16))):
    d = sub[sub.roi == roi_lab].set_index("predictor").reindex(PRED_ORDER)
    lo, hi = zip(*[_ci(r, n) for r, n in zip(d["r"], d["n"])])
    ax.errorbar(d["r"], ys + off, xerr=[d["r"] - np.array(lo), np.array(hi) - d["r"]],
                fmt="o", ms=5, color=ROI_COLOR[roi_lab], ecolor=ROI_COLOR[roi_lab],
                elinewidth=1.2, capsize=2, label=roi_lab,
                markeredgecolor="black", markeredgewidth=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_yticks(ys); ax.set_yticklabels([PRED_LABEL[p] for p in PRED_ORDER], fontsize=8)
ax.set_xlabel(f"correlation with caching index [{HEADLINE}]", fontsize=9)
ax.set_title("A   overlap vs control predictors", fontsize=9, loc="left")
ax.legend(fontsize=8, frameon=False, loc="lower right")
sns.despine(ax=ax)

# ---- B: overlap across caching measures ----
ax = axes[1]
ov = CTRL[CTRL.predictor == "sq_overlap"]
ms = [m for m in CACHE_MEASURES if m in set(ov.measure)]
ys = np.arange(len(ms))[::-1]
for roi_lab, off in zip(ROI_COLOR, (+0.16, -0.16)):
    d = ov[ov.roi == roi_lab].set_index("measure").reindex(ms)
    lo, hi = zip(*[_ci(r, n) for r, n in zip(d["r"], d["n"])])
    ax.errorbar(d["r"], ys + off, xerr=[d["r"] - np.array(lo), np.array(hi) - d["r"]],
                fmt="o", ms=5, color=ROI_COLOR[roi_lab], ecolor=ROI_COLOR[roi_lab],
                elinewidth=1.2, capsize=2, markeredgecolor="black",
                markeredgewidth=0.5)
ax.axvline(0, color="gray", ls="--", lw=0.8)
ax.set_yticks(ys); ax.set_yticklabels(ms, fontsize=8)
ax.set_xlabel("correlation of overlap\nwith caching index", fontsize=9)
ax.set_title("B   which behavioural index?", fontsize=9, loc="left")
sns.despine(ax=ax)

# no tight_layout here: the gridspec already sets the spacing, and the two
# conflict (matplotlib warns). bbox_inches="tight" at save handles the margins.
for ext in ("pdf", "png"):
    _p = os.path.join(cfg.fig_dir, f"fig_overlap_controls_{cfg.event}.{ext}")
    fig.savefig(_p, dpi=300, bbox_inches="tight")
    print(f"saved {_p}")
plt.show()


# %% reliability figure
# How well does each vector replicate across the odd/even split? This is the
# precondition for everything else: an overlap between two vectors that do not
# themselves replicate is not interpretable. Plotted as the raw split-half
# correlation, so 0 means the two halves agree no better than chance and 1
# means they agree perfectly.
REL_COLS = [("rel_rule", "rule"), ("rel_move", "movement")]
REL_COLOR = {"rule": "#333333", "movement": "#1B7837"}

fig, ax = plt.subplots(figsize=(max(78, 42 * len(roi_order)) * MM, 66 * MM))
rng = np.random.default_rng(0)
w = 0.34
xs = np.arange(len(roi_order))

for k, (col, nm) in enumerate(REL_COLS):
    off = (k - 0.5) * w
    for xi, roi in zip(xs, roi_order):
        v = df.loc[df.roi == roi, col].dropna().to_numpy()
        ax.scatter(xi + off + rng.normal(0, 0.045, len(v)), v, s=13,
                   color=REL_COLOR[nm], alpha=0.35, linewidths=0, zorder=2)
        ax.errorbar(xi + off, v.mean(), yerr=stats.sem(v), fmt="o",
                    color=REL_COLOR[nm], markersize=4.5, capsize=3,
                    linewidth=1.2, zorder=4,
                    markeredgecolor="black", markeredgewidth=0.5)
    ax.scatter([], [], color=REL_COLOR[nm], s=22, label=f"{nm} vector")

ax.axhline(0, color="gray", ls="--", lw=0.8, zorder=1)
ax.set_xticks(xs)
ax.set_xticklabels([label[r] for r in roi_order], fontsize=9)
ax.set_ylabel("Split-half reliability, cosine\n(odd vs even runs)", fontsize=9)
ax.set_title("Do the vectors replicate across CV splits?", fontsize=10, pad=8)
ax.legend(fontsize=8, frameon=False, loc="upper right")
sns.despine(ax=ax, trim=True)
plt.tight_layout()

for ext in ("pdf", "png"):
    _p = os.path.join(cfg.fig_dir, f"fig_vector_reliability_{cfg.event}.{ext}")
    fig.savefig(_p, dpi=300, bbox_inches="tight")
    print(f"saved {_p}")
plt.show()

print("\nMean split-half reliability")
_cols = ["rel_rule", "rel_move", "rel_rule_spearman", "rel_move_spearman",
         "mag_rule", "mag_move"]
_cols = [c for c in _cols if c in df.columns]
print(df[df.roi.isin(roi_order)].groupby("roi")[_cols]
      .mean().rename(index=label).round(3).to_string())
print("  rel_* are COSINES — these are the attenuation factors, and")
print("  cos_attn = cos_cv * sqrt(rel_rule * rel_move) holds exactly.")
print("  The _spearman columns centre each vector first, so they describe a")
print("  slightly different vector; shown for reference only.")


# %% split-half reliability of sq_overlap itself (four-way run split)
# This is the quantity Sam's rule of thumb refers to: the brain-behaviour
# correlation should not exceed the reliability of the measure entering it.
# sq_overlap is a product of two cosines, so the reliability of a single cosine
# (below) is only a proxy — this is the direct version, from two sq_overlap
# estimates computed on disjoint halves of the runs.
if {"sq_half_a", "sq_half_b"} <= set(df.columns):
    print("\nSplit-half reliability of sq_overlap (disjoint run halves)")
    print(f"{'ROI':>8} {'r':>8} {'p':>8} {'rho':>8} {'p':>8} {'Spearman-Brown':>15}")
    for roi in roi_order:
        d = df[df.roi == roi].dropna(subset=["sq_half_a", "sq_half_b"])
        if len(d) < 4:
            continue
        r, pv = pearsonr(d["sq_half_a"], d["sq_half_b"])
        rho, prho = spearmanr(d["sq_half_a"], d["sq_half_b"])
        sb = 2*r/(1+r) if r > -1 else np.nan
        print(f"{label[roi]:>8} {r:>8.3f} {pv:>8.3f} {rho:>8.3f} {prho:>8.3f} "
              f"{sb:>15.3f}")
    print("  Spearman-Brown steps up from half-length runs to the full measure.")
    print("  Compare against the overlap-caching correlation: if the correlation")
    print("  is LARGER than this, the measure cannot support it.")
else:
    print("\n[skip] sq_half_a / sq_half_b not in the CSV — rerun the subject script")


# %% reliability of a single cosine estimate (proxy)
# rel_rule / rel_move describe the VECTORS. What the caching correlation
# actually depends on is whether a subject's OVERLAP estimate replicates.
#
# cos1 and cos2 are two independent estimates of the same subject's cosine
# (disjoint halves of the data), so their correlation across subjects IS that
# reliability. If it is near zero, sq_overlap carries no stable between-subject
# signal and any correlation with behaviour is a coincidence — regardless of
# how the vectors themselves look.
#
# Spearman-Brown steps up from a half-length estimate to the full measure.
print("\nReliability of the per-subject overlap estimate")
# Pearson and Spearman both reported: these are scalars across 27 subjects,
# not beta patterns, so Pearson is the natural choice — but at this n a couple
# of points can carry it, and Spearman says whether that is happening.
print(f"{'ROI':>8} {'r(cos1,cos2)':>13} {'p':>8} {'rho':>8} {'p':>8} "
      f"{'Spearman-Brown':>15}")
for roi in roi_order:
    d = df[df.roi == roi].dropna(subset=["cos1", "cos2"])
    r, pv = pearsonr(d["cos1"], d["cos2"])
    rho, prho = spearmanr(d["cos1"], d["cos2"])
    sb = 2*r / (1 + r) if r > -1 else np.nan
    print(f"{label[roi]:>8} {r:>13.3f} {pv:>8.3f} {rho:>8.3f} {prho:>8.3f} "
          f"{sb:>15.3f}")
print("  near 0 -> sq_overlap does not replicate across subjects, and any")
print("  correlation with behaviour should be treated as chance")

fig, axes = plt.subplots(1, len(roi_order),
                         figsize=(58*len(roi_order)*MM, 60*MM), squeeze=False)
for ax, roi in zip(axes[0], roi_order):
    d = df[df.roi == roi].dropna(subset=["cos1", "cos2"])
    r, pv = pearsonr(d["cos1"], d["cos2"])
    ax.scatter(d["cos1"], d["cos2"], s=18, color=BAR_COLOR,
               edgecolor="black", linewidths=0.3)
    lim = np.abs(np.r_[d["cos1"], d["cos2"]]).max() * 1.15
    ax.plot([-lim, lim], [-lim, lim], color="0.75", lw=0.8, ls="--")
    ax.axhline(0, color="0.9", lw=0.6); ax.axvline(0, color="0.9", lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(f"{label[roi]}   r = {r:.2f} (p = {pv:.3f})", fontsize=8)
    ax.set_xlabel("cos (rule odd, move even)", fontsize=7.5)
    ax.set_ylabel("cos (rule even, move odd)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    sns.despine(ax=ax)
plt.tight_layout()
for ext in ("pdf", "png"):
    _p = os.path.join(cfg.fig_dir, f"fig_overlap_reliability_{cfg.event}.{ext}")
    fig.savefig(_p, dpi=300, bbox_inches="tight")
    print(f"saved {_p}")
plt.show()


# %% summary: correlation against both kinds of reliability
# The two controls answer different objections and are easy to conflate:
#
#   PER-SUBJECT vector reliability (rel_rule, rel_move) enters as a covariate.
#   The partial asks whether the correlation survives holding constant how well
#   each subject's vectors were estimated. It rules out data quality driving it.
#
#   MEASURE-LEVEL reliability of sq_overlap is one number for the sample, not a
#   covariate. It bounds how large ANY correlation with sq_overlap can be:
#   r_observed <= sqrt(rel_measure * rel_behaviour). This is the check Sam
#   described — a correlation larger than the reliability is a problem.
HEAD = "rt"
print(f"\n{'='*66}\nSUMMARY: overlap-caching correlation vs its reliability "
      f"[{HEAD}]\n{'='*66}")

for roi in roi_order:
    d = df[df.roi == roi]
    ci = caching_index(HEAD)
    m = pd.concat([ci, d.set_index("subject")[["sq_overlap", "rel_rule",
                                               "rel_move"]]],
                  axis=1, join="inner").dropna()
    r_obs, p_obs = pearsonr(m["sq_overlap"], m["caching_index"])

    # measure-level reliability: four-way split if available, else the proxy
    dd = d.dropna(subset=["sq_half_a", "sq_half_b"]) \
        if {"sq_half_a", "sq_half_b"} <= set(df.columns) else pd.DataFrame()
    if len(dd) >= 4:
        r_half = pearsonr(dd["sq_half_a"], dd["sq_half_b"])[0]
        src = "four-way split"
    else:
        dd = d.dropna(subset=["cos1", "cos2"])
        r_half = pearsonr(dd["cos1"], dd["cos2"])[0]
        src = "r(cos1,cos2) proxy"
    rel_meas = 2*r_half/(1+r_half) if r_half > -1 else np.nan

    print(f"\n{label[roi]}")
    print(f"  overlap-caching correlation        r = {r_obs:+.3f}  (p = {p_obs:.3f})")
    print(f"  per-subject vector reliability     rule {d.rel_rule.mean():.3f}, "
          f"move {d.rel_move.mean():.3f}   (cosine, used as covariates)")
    print(f"  measure-level reliability          {rel_meas:.3f}   "
          f"({src}, Spearman-Brown)")
    if np.isfinite(rel_meas) and rel_meas > 0:
        ceiling = np.sqrt(rel_meas)
        print(f"  ceiling if behaviour were perfect  r <= {ceiling:.3f}   "
              f"-> observed is {'BELOW' if abs(r_obs) <= ceiling else 'ABOVE'} it")
    else:
        print(f"  ceiling                            undefined "
              f"(reliability <= 0)")


# %% robustness of the brain-behaviour correlation
# Three checks:
#   1. CEILING — the largest correlation the measure could produce, given its
#      own reliability and assuming the behavioural index is noise-free. The
#      observed value should not exceed it.
#   2. OUTLIERS — Spearman, plus a bootstrap CI, in case a couple of subjects
#      carry the Pearson.
#   3. HALF-DATA — recompute the correlation using an overlap estimate from
#      only half the runs. If it is roughly similar, the effect is not an
#      artefact of how the full estimate was assembled.
N_BOOT_CORR = 10000
HEAD = "rt"
_rng = np.random.default_rng(0)
ci_head = caching_index(HEAD)

print(f"\n{'='*70}\nROBUSTNESS: overlap-caching correlation [{HEAD}]\n{'='*70}")
print("Read the half-data rows first: if a correlation is real, overlap from")
print("each half of the runs should relate to caching the SAME way. Opposite")
print("signs or one half near zero means the full-data value is not stable.")
for roi in roi_order:
    d = df[df.roi == roi].set_index("subject")
    m = pd.concat([ci_head, d[["sq_overlap", "cos1", "cos2"] +
                              [c for c in ("sq_half_a", "sq_half_b")
                               if c in df.columns]]],
                  axis=1, join="inner").dropna(subset=["caching_index",
                                                       "sq_overlap"])
    x = m["sq_overlap"].to_numpy(); y = m["caching_index"].to_numpy()
    r_p, p_p = pearsonr(x, y)                      # two-tailed
    r_s, p_s = spearmanr(x, y)
    p_p_1t = p_p/2 if ((r_p > 0) == (CORR_DIRECTION == "greater")) else 1 - p_p/2

    # bootstrap over subjects — Pearson and Spearman, same resamples
    idx = _rng.integers(0, len(x), (N_BOOT_CORR, len(x)))
    boots = np.array([pearsonr(x[i], y[i])[0] for i in idx])
    boots_s = np.array([spearmanr(x[i], y[i])[0] for i in idx])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lo_s, hi_s = np.percentile(boots_s[np.isfinite(boots_s)], [2.5, 97.5])

    # ceiling from the measure's own reliability
    rel_raw = pearsonr(m["cos1"], m["cos2"])[0]
    rel_sb = 2*rel_raw/(1+rel_raw) if rel_raw > -1 else np.nan

    print(f"\n{label[roi]}  (n = {len(m)})")
    print(f"  Pearson   r = {r_p:+.3f}  p = {p_p:.3f} (two-tailed)"
          f"   one-tailed = {p_p_1t:.3f}")
    print(f"  Spearman  rho = {r_s:+.3f}  p = {p_s:.3f}"
          f"   {'(holds up)' if abs(r_s) > 0.6*abs(r_p) else '(DROPS — leverage)'}")
    print(f"  bootstrap 95% CI  r   [{lo:+.3f}, {hi:+.3f}]"
          f"   excludes 0: {lo > 0 or hi < 0}")
    print(f"  bootstrap 95% CI  rho [{lo_s:+.3f}, {hi_s:+.3f}]"
          f"   excludes 0: {lo_s > 0 or hi_s < 0}")
    for nm, rel in (("raw", rel_raw), ("Spearman-Brown", rel_sb)):
        if np.isfinite(rel) and rel > 0:
            c = np.sqrt(rel)
            print(f"  ceiling ({nm:15s}) r <= {c:.3f}"
                  f"   observed is {'ABOVE' if abs(r_p) > c else 'at/below'}")
        else:
            print(f"  ceiling ({nm:15s}) undefined (reliability <= 0)")

    # same correlation from half the runs
    for col in ("sq_half_a", "sq_half_b"):
        if col in m.columns and m[col].notna().sum() >= 6:
            mm = m.dropna(subset=[col])
            rh, ph = pearsonr(mm[col], mm["caching_index"])
            rs_h, ps_h = spearmanr(mm[col], mm["caching_index"])
            print(f"  half-data ({col})  r = {rh:+.3f}  p = {ph:.3f}   "
                  f"rho = {rs_h:+.3f}  p = {ps_h:.3f}")
# %%