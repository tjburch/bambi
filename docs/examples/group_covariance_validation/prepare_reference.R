# Small crossed fixture and independent cell indexing for reference comparisons.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: prepare_reference.R bernoulli|binomial output-dir seed")
family <- match.arg(args[1], c("bernoulli", "binomial"))
seed <- suppressWarnings(as.integer(args[3]))
if (is.na(seed) || seed < 1L) stop("seed must be a positive integer")
if (!requireNamespace("jsonlite", quietly = TRUE)) stop("jsonlite is required")
if (file.exists(args[2])) stop("Output path already exists; use a separate run directory")
set.seed(seed)
data <- expand.grid(
  subject = paste0("s", seq_len(6L)), condition = c("a", "b"),
  context = c("c", "d"), year = c(0L, 1L, 3L), replicate = seq_len(2L),
  KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE
)
# Preserve lower-order combinations while withholding one three-way group.
data <- subset(data, !(subject == "s1" & condition == "a" & context == "c"))
data <- subset(data, !(subject == "s2" & year == 1L))
rownames(data) <- NULL
# A binary grid keeps the fixed design identical across R, Python and Stan parsers.
data$x1 <- round(rnorm(nrow(data)) * 8) / 8
data$x2 <- rbinom(nrow(data), 1L, 0.5)
data$one <- 1
data$visit <- factor(data$year, levels = c(0L, 1L, 3L))
data$trials <- if (family == "bernoulli") rep(1L, nrow(data)) else sample(2:8, nrow(data), TRUE)
beta <- c(-0.3, 0.4, -0.2)
scales <- c(0.4, 0.25, 0.2, 0.15)
correlations <- c(0.6, -0.35, 0.2, 0.45)
group_columns <- list("subject", c("subject", "condition"), c("subject", "context"),
                      c("subject", "condition", "context"))
cells <- list()
reference <- list()
row_cell <- matrix(NA_integer_, nrow(data), 4L)
coefficient <- numeric()
cell_offset <- 0L
group_offset <- 0L
for (b in seq_len(4L)) {
  group <- do.call(interaction, c(data[group_columns[[b]]], list(drop = TRUE, lex.order = TRUE)))
  group <- as.integer(group)
  current <- unique(data.frame(group = group, time = data$year))
  current <- current[order(current$group, current$time), , drop = FALSE]
  rownames(current) <- NULL
  key <- paste(current$group, current$time, sep = ":")
  row_cell[, b] <- match(paste(group, data$year, sep = ":"), key) + cell_offset
  current$previous <- c(0L, head(seq_len(nrow(current)), -1L))
  first <- !duplicated(current$group)
  current$previous[first] <- 0L
  effects <- numeric(nrow(current))
  for (i in seq_len(nrow(current))) {
    p <- current$previous[i]
    if (p == 0L) {
      effects[i] <- scales[b] * rnorm(1L)
    } else {
      persistence <- correlations[b]^(current$time[i] - current$time[p])
      effects[i] <- persistence * effects[p] + scales[b] * sqrt(1 - persistence^2) * rnorm(1L)
    }
  }
  labels <- paste0("b", b, "cell", seq_len(nrow(current)))
  K <- correlations[b]^abs(outer(current$time, current$time, "-"))
  K <- K * outer(current$group, current$group, "==")
  dimnames(K) <- list(labels, labels)
  reference[[paste0("K", b)]] <- K
  data[[paste0("cell", b)]] <- labels[row_cell[, b] - cell_offset]
  current$previous[current$previous > 0L] <- current$previous[current$previous > 0L] + cell_offset
  current$group <- current$group + group_offset
  current$block <- b
  cells[[b]] <- current
  coefficient <- c(coefficient, effects)
  cell_offset <- cell_offset + nrow(current)
  group_offset <- max(current$group)
}
cells <- do.call(rbind, cells)
X <- as.matrix(data[c("one", "x1", "x2")])
eta <- drop(X %*% beta) + rowSums(matrix(coefficient[row_cell], nrow(data), 4L))
data$y <- rbinom(nrow(data), data$trials, plogis(eta))
reference$data <- data
reference$fixed_rho <- correlations
stan <- list(
  N = nrow(data), P = ncol(X), X = X, trials = data$trials, y = data$y,
  C = nrow(cells), G = max(cells$group), block_id = cells$block,
  group_id = cells$group, time = cells$time, previous = cells$previous,
  row_cell = row_cell, prior_only = 0L
)
dir.create(args[2], recursive = TRUE)
saveRDS(reference, file.path(args[2], "input.rds"))
saveRDS(list(beta = beta, sd = scales, rho = correlations, coefficient = coefficient,
             probability = plogis(eta), seed = seed, family = family),
        file.path(args[2], "truth.rds"))
write.csv(data, file.path(args[2], "data.csv"), row.names = FALSE)
write.csv(cells, file.path(args[2], "cells.csv"), row.names = FALSE)
jsonlite::write_json(stan, file.path(args[2], "data.json"), auto_unbox = TRUE, digits = 17)
