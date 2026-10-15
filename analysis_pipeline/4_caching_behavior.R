#!/usr/bin/env Rscript
# Behavioral caching (Fig 5b) and overall RT, matched vs unmatched trials (critical = matched).

data_dir <- Find(dir.exists, c(file.path("..", "data"), "data"))   # run from analysis_pipeline/ or the repo root
b <- read.csv(file.path(data_dir, "behavior.csv"))
b <- b[!is.na(b$rt), ]
show <- function(label, tt) cat(label, ": t =", tt$statistic, ", df =", tt$parameter, ", p =", tt$p.value, "\n")

# |RT_reflection - RT_rotation| per participant and trial class, unmatched vs matched (Fig 5b)
cell <- aggregate(rt ~ subject + critical + manipulation_type, data = b, FUN = mean)
ctx <- reshape(cell, idvar = c("subject", "critical"), timevar = "manipulation_type", direction = "wide")
ctx$delta <- abs(ctx$rt.mirror - ctx$rt.rotation)
d <- reshape(ctx[, c("subject", "critical", "delta")], idvar = "subject", timevar = "critical", direction = "wide")
show("caching, unmatched - matched", t.test(d$delta.uncritical, d$delta.critical, paired = TRUE))

# overall RT, unmatched vs matched
rt <- aggregate(rt ~ subject + critical, data = b, FUN = mean)
rt <- reshape(rt, idvar = "subject", timevar = "critical", direction = "wide")
show("overall RT, unmatched - matched", t.test(rt$rt.uncritical, rt$rt.critical, paired = TRUE))
