# analysis_driver.py
# -----------------------------------------------------------------------------
# Cells for analysis.ipynb. Paste in, or %run -i from a notebook cell.
#
# The rule: this file DEFINES NOTHING about the analysis. Every subject list,
# threshold, ROI name and path comes from pipeline_config.py, so the notebook
# can't drift from the scripts it drives. It has two jobs:
#
#   1. submit subject-level scripts to sbatch (these load full trial data and
#      belong on a compute node, not in a notebook kernel)
#   2. run group-level scripts inline (small inputs, fast, worth iterating on)
#
# Each stage below is independent; run only the cell you need.
# -----------------------------------------------------------------------------

# %% setup
import os
import sys
import subprocess
from pathlib import Path

sys.path.insert(0, "/home/sk2949/project/github/mrdecode/analysis/fmri/glm_mvpa")
from pipeline_config import cfg, SUBJECT_LIST, ROI_SETS, roi_strings, cli_args

os.chdir(cfg.glm_mvpa_dir)
print(f"{len(SUBJECT_LIST)} subjects | event={cfg.event} | "
      f"thresholds {cfg.run_censor_thr}/{cfg.subj_runs_thr}")


# Partitions to spread subject-level jobs across. `sbatch -p` on the command
# line overrides the #SBATCH -p line inside each wrapper, so this is the only
# place partitions are chosen. Jobs alternate through the list in order.
PARTITIONS = ["psych_day", "psych_week"]


def submit(sh_script, *args, partition=None, dry_run=False):
    """sbatch a wrapper with positional args. dry_run prints instead."""
    cmd = ["sbatch"] + (["-p", partition] if partition else []) \
          + [sh_script, *map(str, args)]
    if dry_run:
        print(" ".join(cmd)); return
    subprocess.run(cmd, check=True)


def submit_spread(sh_script, items, arg_fn, dry_run=False):
    """Submit one job per item, cycling through PARTITIONS."""
    for i, item in enumerate(items):
        submit(sh_script, *arg_fn(item),
               partition=PARTITIONS[i % len(PARTITIONS)], dry_run=dry_run)


def missing_outputs(pattern_fn, items):
    """Return the items whose expected output doesn't exist yet, so a
    stage can be resumed without resubmitting finished work."""
    return [i for i in items if not Path(pattern_fn(i)).exists()]


# %% stage 0: masks (run once; interactive, no data loading)
# Julich cortical/hippocampal masks come from juelich_showcase.ipynb.
# Striatal masks from Harvard-Oxford, warped into 2009cAsym:
#   %run -i build_harvardoxford_striatum_masks.py
#
# Sanity: every ROI the config knows about should have a mask file.
for roi in roi_strings(list(ROI_SETS)):
    p = os.path.join(cfg.atlas_dir, f"{roi}.nii.gz")
    print(f"{'ok     ' if os.path.exists(p) else 'MISSING'} {roi}")


# %% stage 1: ROI RSA (subject_rsa_full.py) — sbatch, one job per ROI
# Writes correlation_matrix_full, betas_absolute/zscored, variance_partitioning
# pkls per subject per ROI. Loops all subjects inside, so one job per ROI.
ROI_STAGE_SETS = ["main", "supp_cortical", "striatum"]   # edit to rerun a subset

submit_spread("rsa_regression/subject_rsa_full.sh", roi_strings(ROI_STAGE_SETS),
              lambda roi: (roi, *cli_args(roi), cfg.atlas),
              dry_run=True)   # set False to actually submit


# %% stage 2: variance partitioning + noise ceiling — after stage 1 completes
# variance_partition.py takes --rois; pass the config's list so it can't drift.
rois = roi_strings(ROI_STAGE_SETS)
subprocess.run([
    "python", "rsa_regression/variance_partition.py",
    "--rsa_dir", cfg.rsa_dir, "--event", cfg.event,
    "--rois", *rois, "--subjects", *SUBJECT_LIST,
    "--predictors", "task_rule", "goal_location", "optimal_movement",
], check=True)

# noise_ceiling.py still hardcodes its roi_dict; until it imports the config,
# make sure that list matches `rois` above before running it.
# %run -i rsa_regression/noise_ceiling.py


# %% stage 3: searchlight (searchlight_rsa.py) — sbatch, one job per subject
todo = missing_outputs(
    lambda s: os.path.join(cfg.searchlight_dir,
        f"searchlight_rsa_task_rule_absolute_{s}_{cfg.event}_{cfg.censor_label}.nii.gz"),
    SUBJECT_LIST)
print(f"searchlight: {len(todo)} subjects still to run")
submit_spread("searchlight/searchlight_rsa.sh", todo,
              lambda s: (s, *cli_args(s)), dry_run=True)


# %% stage 4: parcel RSA (parcel_rsa_subject.py) — sbatch, one job per subject
todo = missing_outputs(
    lambda s: os.path.join(cfg.parcel_dir,
        f"parcel_rsa_{cfg.parcel_tag}_{s}_{cfg.event}_{cfg.censor_label}.npz"),
    SUBJECT_LIST)
print(f"parcel rsa: {len(todo)} subjects still to run")
submit_spread("parcel/parcel_rsa_subject.sh", todo,
              lambda s: (s, *cli_args(s)), dry_run=True)


# %% stage 5: group-level analyses — interactive
# These read small per-subject outputs and are quick; run them here and look
# at the results rather than submitting them.
#
# %run -i rsa_regression/rsa_model_comparison.py
# %run -i rsa_regression/rsa_distance_comparison.py
# %run -i parcel/parcel_rsa_group.py
# %run -i searchlight/searchlight_striatum_figure.py


# %% stage 6: figures — interactive
# %run -i figures/figure2_generate.py
# %%