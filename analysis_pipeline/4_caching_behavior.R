#!/usr/bin/env Rscript
# Behavioral caching, Fig 5b, plus the overall-RT sentence in Results.
# Note: critical = MATCHED, uncritical = UNMATCHED.

# data/ sits beside this folder; run from here or from the repo root.
data_dir <- Find(function(d) file.exists(file.path(d, "behavior.csv")),
                 c(file.path("..", "data"), "data"))
if (is.null(data_dir)) stop("Cannot find data/: run from analysis_pipeline/ or the repository root.")
behavior_csv <- file.path(data_dir, "behavior.csv")

# helpers
fmt_p <- function(p) {
  # same as fmt_p in config.py
  if (p < 0.001) return("p < .001")
  s <- if (p < 0.01 || (p >= 0.045 && p < 0.055)) sprintf("%.3f", p) else sprintf("%.2f", p)
  paste0("p = ", sub("^0\\.", ".", s))
}

# paired t-test, reported as in the manuscript
report_paired <- function(x, y, label) {
  tt <- t.test(x, y, paired = TRUE)
  cat(sprintf("  %s: t(%d) = %.2f, %s\n",
              label, tt$parameter, tt$statistic, fmt_p(tt$p.value)))
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

cat("Behavioral caching analyses\n")

# (1) caching contrast (Fig 5b): |RT_ref - RT_rot| within each trial type
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

cat("\n(1) Caching contrast: |RT_reflection - RT_rotation|, Fig 5b\n")
report_paired(d1$uncritical, d1$critical, "unmatched - matched")

# (2) overall RT
d2 <- wide_means(b, "rt", "critical")
d2 <- d2[complete.cases(d2), ]

cat("\n(2) Overall RT, collapsing across context\n")
report_paired(d2$uncritical, d2$critical, "unmatched - matched")

cat("\n")
