#!/usr/bin/env Rscript
# Behavioral results, Fig 2. Base R only.

# data/ sits beside this folder; run from here or from the repo root.
data_dir <- Find(function(d) file.exists(file.path(d, "behavior.csv")),
                 c(file.path("..", "data"), "data"))
if (is.null(data_dir)) stop("Cannot find data/: run from analysis_pipeline/ or the repository root.")
behavior_csv <- file.path(data_dir, "behavior.csv")

b <- read.csv(behavior_csv, stringsAsFactors = FALSE)

fmt_p <- function(p) {
  # same as fmt_p in config.py
  if (p < 0.001) return("p < .001")
  s <- if (p < 0.01 || (p >= 0.045 && p < 0.055)) sprintf("%.3f", p) else sprintf("%.2f", p)
  paste0("p = ", sub("^0\\.", ".", s))
}

# Rotation vs Reflection, paired, on a subject x condition table.
paired_line <- function(w, label) {
  w <- w[complete.cases(w), ]
  tt <- t.test(w$rotation, w$mirror, paired = TRUE)
  cat(sprintf("  %-22s n = %2d | Rot %8.3f  Ref %8.3f | t(%d) = %5.2f, %s\n",
              label, nrow(w), mean(w$rotation), mean(w$mirror),
              tt$parameter, tt$statistic, fmt_p(tt$p.value)))
  invisible(tt)
}

# subject x condition means of `value`, wide
wide_by_condition <- function(d, value) {
  a <- aggregate(d[[value]],
                 by = list(subject = d$subject,
                           manipulation_type = d$manipulation_type),
                 FUN = mean, na.rm = TRUE)
  names(a)[3] <- "value"
  w <- reshape(a, idvar = "subject", timevar = "manipulation_type",
               direction = "wide")
  names(w) <- sub("^value\\.", "", names(w))
  w
}

cat("=====================================================================\n")
cat("Behavioral results: Figure 2\n")
cat("=====================================================================\n")
cat(sprintf("  %d trials, %d subjects\n\n",
            nrow(b), length(unique(b$subject))))

# trial-level table with the four error measures
cell <- aggregate(
  cbind(initial_abs = abs(b$initial_heading_deg),
        initial = b$initial_heading_deg,
        endpoint_abs = abs(b$endpoint_angle_deg),
        endpoint = b$endpoint_angle_deg) ~
    subject + trial + manipulation_type + critical,
  data = b, FUN = mean, na.rm = TRUE, na.action = na.pass)

# (1) error magnitude
cat("(1) Average absolute initial error\n")
w <- wide_by_condition(cell, "initial_abs")
w <- w[complete.cases(w), ]
cat(sprintf("  Rotation   %.2f +/- %.2f\n", mean(w$rotation), sd(w$rotation)))
cat(sprintf("  Reflection %.2f +/- %.2f\n\n", mean(w$mirror), sd(w$mirror)))

# (2) Rotation vs Reflection, four measures (Fig 2d)
cat("(2) Rotation vs Reflection, by error measure (Fig 2d)\n")
paired_line(wide_by_condition(cell, "initial_abs"), "absolute initial")
paired_line(wide_by_condition(cell, "initial"), "signed initial")
paired_line(wide_by_condition(cell, "endpoint_abs"), "absolute endpoint")
paired_line(wide_by_condition(cell, "endpoint"), "signed endpoint")
cat("\n")

# (3) learning slope (Fig 2e): ABSOLUTE initial error vs trial index within condition
cat("(3) Learning slope across scanning, vs zero (Fig 2e)\n")
slopes <- data.frame()
for (s in unique(cell$subject)) {
  for (m in unique(cell$manipulation_type)) {
    d <- cell[cell$subject == s & cell$manipulation_type == m, ]
    d <- d[order(d$trial), ]
    d <- d[!is.na(d$initial_abs), ]
    if (nrow(d) < 3) next
    d$trial_within_type <- seq_len(nrow(d))
    cf <- coef(lm(initial_abs ~ trial_within_type, data = d))
    slopes <- rbind(slopes, data.frame(subject = s, manipulation_type = m,
                                       slope = cf[["trial_within_type"]]))
  }
}
for (m in c("rotation", "mirror")) {
  v <- slopes$slope[slopes$manipulation_type == m]
  tt <- t.test(v, mu = 0)
  cat(sprintf("  %-22s n = %2d | mean %8.5f | t(%d) = %5.2f, %s\n",
              ifelse(m == "mirror", "Reflection", "Rotation"), length(v),
              mean(v), tt$parameter, tt$statistic, fmt_p(tt$p.value)))
}
cat("\n")

# (4) RT and MT (Fig 2f), straight from the trial rows
cat("(4) Reaction and movement time, Rotation vs Reflection (Fig 2f)\n")
paired_line(wide_by_condition(b, "rt"), "RT (s)")
paired_line(wide_by_condition(b, "mt"), "MT (s)")
cat("\n")

# (5) tutorial phase (Fig 2b-c), n = 24 (three subjects have no tutorial data)

tu <- read.csv(file.path(data_dir, "tutorial.csv"), stringsAsFactors = FALSE)

# `trial` runs 1-32 across both blocks; this is the index within a block
tu$trial_in_block <- tu$trial - 16 * (tu$block - 1)

# flip Reflection errors at unmatched targets (T2, T4) so signs line up
flip <- tu$manipulation_type == "mirror" & tu$which_target %in% c(2, 4)
tu$err <- ifelse(flip, -tu$initial_heading_deg, tu$initial_heading_deg)

cat("(5) Tutorial learning phase (Fig 2b-c)\n")
cat(sprintf("  %d trials, %d subjects\n\n", nrow(tu), length(unique(tu$subject))))

cat("  Learning: first two vs last two trials of each block (Fig 2b)\n")
for (m in c("rotation", "mirror")) {
  d <- tu[tu$manipulation_type == m, ]
  d <- d[order(d$subject, d$trial_in_block), ]
  first <- aggregate(err ~ subject, data = do.call(rbind,
             by(d, d$subject, function(g) head(g, 2))), FUN = mean)
  last  <- aggregate(err ~ subject, data = do.call(rbind,
             by(d, d$subject, function(g) tail(g, 2))), FUN = mean)
  w <- merge(first, last, by = "subject", suffixes = c("_first", "_last"))
  tt <- t.test(w$err_first, w$err_last, paired = TRUE)
  cat(sprintf("    %-11s n = %2d | first %7.2f  last %6.2f | t(%d) = %5.2f, %s\n",
              ifelse(m == "mirror", "Reflection", "Rotation"), nrow(w),
              mean(w$err_first), mean(w$err_last),
              tt$parameter, tt$statistic, fmt_p(tt$p.value)))
}

cat("\n  Asymptote: slope over trials 9-16 of each block, vs zero (Fig 2c)\n")
for (m in c("rotation", "mirror")) {
  d <- tu[tu$manipulation_type == m & tu$trial_in_block >= 9, ]
  sl <- sapply(split(d, d$subject), function(g)
    if (nrow(g) > 2) coef(lm(err ~ trial_in_block, data = g))[2] else NA)
  sl <- sl[!is.na(sl)]
  tt <- t.test(sl, mu = 0)
  cat(sprintf("    %-11s n = %2d | mean slope %+6.3f | t(%d) = %5.2f, %s\n",
              ifelse(m == "mirror", "Reflection", "Rotation"), length(sl),
              mean(sl), tt$parameter, tt$statistic, fmt_p(tt$p.value)))
}
cat("\n")
