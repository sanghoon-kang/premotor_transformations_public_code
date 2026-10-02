#!/usr/bin/env Rscript
# Behavioral caching — Figure 5b, plus the overall-RT sentence in Results.
# Base R only, nothing to install.
# Run from the repository root:  Rscript 4_caching_behavior.R
#
# NOTE: in `critical`, the level "critical" means MATCHED and "uncritical"
# means UNMATCHED. Historical naming, kept so the data matches the code.

behavior_csv <- file.path("data", "behavior.csv")

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

fmt_p <- function(p) {
  # APA: no leading zero (p cannot exceed 1); 2 decimals, 3 below .01 and
  # within .005 of .05, floored at .001. Matches fmt_p in config.py.
  if (p < 0.001) return("p < .001")
  s <- if (p < 0.01 || (p >= 0.045 && p < 0.055)) sprintf("%.3f", p) else sprintf("%.2f", p)
  paste0("p = ", sub("^0\\.", ".", s))
}

# Paired t-test reported the way the manuscript reports it.
report_paired <- function(x, y, label) {
  d <- x - y
  n <- length(d)
  tt <- t.test(x, y, paired = TRUE)
  cat(sprintf("\n%s\n", label))
  cat(sprintf("  n          = %d\n", n))
  cat(sprintf("  mean diff  = %.4f s (SEM %.4f)\n",
              mean(d), sd(d) / sqrt(n)))
  cat(sprintf("  95%% CI     = [%.4f, %.4f]\n",
              tt$conf.int[1], tt$conf.int[2]))
  cat(sprintf("  Cohen's dz = %.2f\n", mean(d) / sd(d)))
  cat(sprintf("  t(%d) = %.2f, %s\n",
              tt$parameter, tt$statistic, fmt_p(tt$p.value)))
  invisible(tt)
}

# Subject x condition means, returned wide with one column per level of `by`.
wide_means <- function(df, value, by) {
  agg <- aggregate(df[[value]],
                   by = list(subject = df$subject, level = df[[by]]),
                   FUN = mean, na.rm = TRUE)
  names(agg)[3] <- "value"
  out <- reshape(agg, idvar = "subject", timevar = "level",
                 direction = "wide")
  names(out) <- sub("^value\\.", "", names(out))
  out
}

b <- read.csv(behavior_csv, stringsAsFactors = FALSE)

b <- b[!is.na(b$rt), ]

cat("=====================================================================\n")
cat("Behavioral caching analyses\n")
cat("=====================================================================\n")
cat(sprintf("  %d trials, %d subjects\n",
            nrow(b), length(unique(b$subject))))
cat("  reminder: critical = MATCHED, uncritical = UNMATCHED\n")

# ---------------------------------------------------------------------------
# (1) caching contrast -- Fig 5b
# ---------------------------------------------------------------------------
# Mean RT per subject x trial type x context, then the absolute difference
# between contexts within each trial type.

cell <- aggregate(rt ~ subject + critical + manipulation_type,
                  data = b, FUN = mean)

ctx <- reshape(cell, idvar = c("subject", "critical"),
               timevar = "manipulation_type", direction = "wide")
names(ctx) <- sub("^rt\\.", "", names(ctx))

ctx$delta <- abs(ctx$mirror - ctx$rotation)

d1 <- reshape(ctx[, c("subject", "critical", "delta")],
              idvar = "subject", timevar = "critical", direction = "wide")
names(d1) <- sub("^delta\\.", "", names(d1))
d1 <- d1[complete.cases(d1), ]

cat("\n---------------------------------------------------------------------\n")
cat("(1) Caching contrast: |RT_reflection - RT_rotation|, Fig 5b\n")
cat("---------------------------------------------------------------------\n")
cat(sprintf("  matched   M = %.4f s\n", mean(d1$critical)))
cat(sprintf("  unmatched M = %.4f s\n", mean(d1$uncritical)))
report_paired(d1$uncritical, d1$critical,
              "unmatched - matched (positive = consistent with caching)")

# ---------------------------------------------------------------------------
# (2) overall RT
# ---------------------------------------------------------------------------

d2 <- wide_means(b, "rt", "critical")
d2 <- d2[complete.cases(d2), ]

cat("\n---------------------------------------------------------------------\n")
cat("(2) Overall RT, collapsing across context\n")
cat("---------------------------------------------------------------------\n")
cat(sprintf("  matched   M = %.4f s\n", mean(d2$critical)))
cat(sprintf("  unmatched M = %.4f s\n", mean(d2$uncritical)))
report_paired(d2$uncritical, d2$critical,
              "unmatched - matched (positive = matched trials faster)")

cat("\n")
