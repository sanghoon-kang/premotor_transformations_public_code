#!/usr/bin/env Rscript
# Behavioral results (Fig 2): scanning-phase error, learning slopes, RT/MT, and tutorial learning.

data_dir <- Find(dir.exists, c(file.path("..", "data"), "data"))   # run from analysis_pipeline/ or the repo root
b <- read.csv(file.path(data_dir, "behavior.csv"))
tu <- read.csv(file.path(data_dir, "tutorial.csv"))

# subject x condition means, one column per condition (rotation, mirror)
by_condition <- function(d, value) {
  a <- aggregate(d[[value]], list(subject = d$subject, cond = d$manipulation_type), mean, na.rm = TRUE)
  w <- reshape(a, idvar = "subject", timevar = "cond", direction = "wide")
  names(w) <- sub("^x\\.", "", names(w))
  w[complete.cases(w), ]
}
show <- function(label, tt) cat(label, ": t =", tt$statistic, ", df =", tt$parameter, ", p =", tt$p.value, "\n")

# trial-type cells with the four error measures
cell <- aggregate(cbind(initial_abs = abs(initial_heading_deg), initial = initial_heading_deg,
                        endpoint_abs = abs(endpoint_angle_deg), endpoint = endpoint_angle_deg) ~
                    subject + trial + manipulation_type + critical,
                  data = b, FUN = mean, na.rm = TRUE, na.action = na.pass)

# absolute initial error, mean and SD across participants
w <- by_condition(cell, "initial_abs")
cat("absolute initial error: rotation", mean(w$rotation), sd(w$rotation),
    "| reflection", mean(w$mirror), sd(w$mirror), "\n")

# Rotation vs Reflection, paired (Fig 2d, f)
for (v in c("initial_abs", "initial", "endpoint_abs", "endpoint")) {
  w <- by_condition(cell, v)
  show(v, t.test(w$rotation, w$mirror, paired = TRUE))
}
for (v in c("rt", "mt")) {
  w <- by_condition(b, v)
  show(v, t.test(w$rotation, w$mirror, paired = TRUE))
}

# learning slope of absolute initial error across scanning, vs 0 (Fig 2e)
slopes <- do.call(rbind, lapply(split(cell, list(cell$subject, cell$manipulation_type)), function(d) {
  d <- d[order(d$trial), ]
  d <- d[!is.na(d$initial_abs), ]
  data.frame(cond = d$manipulation_type[1], slope = coef(lm(d$initial_abs ~ seq_len(nrow(d))))[2])
}))
for (m in c("rotation", "mirror")) show(paste("scanning slope", m), t.test(slopes$slope[slopes$cond == m]))

# tutorial (Fig 2b-c); reflection errors at unmatched targets (2, 4) sign-flipped
tu$trial_in_block <- tu$trial - 16 * (tu$block - 1)
tu$err <- ifelse(tu$manipulation_type == "mirror" & tu$which_target %in% c(2, 4),
                 -tu$initial_heading_deg, tu$initial_heading_deg)
for (m in c("rotation", "mirror")) {
  d <- tu[tu$manipulation_type == m, ]
  d <- d[order(d$subject, d$trial_in_block), ]
  first <- sapply(split(d$err, d$subject), function(e) mean(head(e, 2)))
  last <- sapply(split(d$err, d$subject), function(e) mean(tail(e, 2)))
  show(paste("tutorial first vs last two trials", m), t.test(first, last, paired = TRUE))
  d <- d[d$trial_in_block >= 9, ]
  sl <- sapply(split(d, d$subject), function(g) coef(lm(err ~ trial_in_block, data = g))[2])
  show(paste("tutorial slope, trials 9-16", m), t.test(sl))
}
