# Native brms reference fits. Each invocation runs one model and saves it first.
Sys.setenv(
  OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1", STAN_NUM_THREADS = "1", MAKEFLAGS = "-j1"
)
options(mc.cores = 1)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop("Usage: reference.R known|us-slopes|us-visits input.rds output-dir prior|posterior chains warmup draws seed")
}
mode <- match.arg(args[1], c("known", "us-slopes", "us-visits"))
phase <- match.arg(args[4], c("prior", "posterior"))
settings <- suppressWarnings(as.integer(args[5:8]))
if (anyNA(settings) || any(settings < 1L)) stop("Sampling settings must be positive integers")
names(settings) <- c("chains", "warmup", "draws", "seed")
if (!requireNamespace("brms", quietly = TRUE) || !requireNamespace("cmdstanr", quietly = TRUE)) {
  stop("Install brms, cmdstanr, and CmdStan in a separate reference environment first")
}
if (file.exists(args[3])) stop("Output path already exists; use a separate run directory")
input <- readRDS(args[2])
identity <- jsonlite::read_json(file.path(dirname(args[2]), paste0("identity-", mode, ".json")),
                               simplifyVector = FALSE)
if (!identical(identity$data_md5, unname(tools::md5sum(file.path(dirname(args[2]), "data.csv"))))) {
  stop("Fixture identity mismatch")
}
data <- input$data
required <- c("y", "trials", "x1", "x2")
if (!is.data.frame(data) || !all(required %in% names(data))) stop("Invalid input$data")
if (anyNA(data[required]) || any(!is.finite(as.matrix(data[required])))) stop("Data must be finite")
if (any(data$trials < 0 | data$trials != floor(data$trials)) ||
    any(data$y < 0 | data$y > data$trials | data$y != floor(data$y))) {
  stop("Invalid binomial outcomes or trial counts")
}
# An explicit constant avoids brms's centered-intercept prior convention.
data$one <- 1
data2 <- list()
if (mode == "known") {
  if (length(input$fixed_rho) != 4L || any(!is.finite(input$fixed_rho)) ||
      any(abs(input$fixed_rho) >= 1)) stop("Provide four valid fixed_rho values")
  grouping_columns <- list("subject", c("subject", "condition"), c("subject", "context"),
                           c("subject", "condition", "context"))
  for (b in seq_len(4L)) {
    cell <- paste0("cell", b)
    key <- paste0("K", b)
    if (!cell %in% names(data) || anyNA(data[[cell]])) stop("Missing cell identifiers")
    data[[cell]] <- factor(data[[cell]])
    K <- input[[key]]
    if (!is.matrix(K) || !is.numeric(K) || anyNA(K) || any(!is.finite(K)) ||
        nrow(K) != ncol(K) || !identical(rownames(K), colnames(K)) ||
        !setequal(rownames(K), levels(data[[cell]]))) stop("Invalid named covariance matrix")
    K <- K[levels(data[[cell]]), levels(data[[cell]]), drop = FALSE]
    if (!isTRUE(all.equal(K, t(K), tolerance = 1e-12)) ||
        any(abs(diag(K) - 1) > 1e-12)) stop("K must be a correlation matrix")
    chol(K)
    representative <- data[match(levels(data[[cell]]), data[[cell]]), , drop = FALSE]
    same_group <- Reduce(`&`, lapply(grouping_columns[[b]], function(name) {
      outer(representative[[name]], representative[[name]], "==")
    }))
    expected <- input$fixed_rho[b]^abs(outer(representative$year, representative$year, "-")) * same_group
    if (!isTRUE(all.equal(unname(K), unname(expected), tolerance = 1e-12))) {
      stop("Known covariance does not match fixed_rho and the grouping/time design")
    }
    data2[[key]] <- K
  }
  formula <- brms::bf(
    y | trials(trials) ~ 0 + one + x1 + x2 +
      (1 | gr(cell1, cov = K1)) + (1 | gr(cell2, cov = K2)) +
      (1 | gr(cell3, cov = K3)) + (1 | gr(cell4, cov = K4))
  )
} else {
  if (!"subject" %in% names(data) || anyNA(data$subject)) stop("Missing subject identifiers")
  data$subject <- factor(data$subject)
  if (mode == "us-slopes") {
    formula <- brms::bf(y | trials(trials) ~ 0 + one + x1 + x2 + (1 + x1 | subject))
  } else {
    if (!"visit" %in% names(data) || anyNA(data$visit)) stop("Missing visit identifiers")
    data$visit <- factor(data$visit)
    formula <- brms::bf(y | trials(trials) ~ 0 + one + x1 + x2 + (0 + visit | subject))
  }
}
# Explicit priors make comparisons independent of brms's data-dependent defaults.
priors <- c(
  brms::set_prior("normal(0, 1.5)", class = "b"),
  brms::set_prior("normal(0, 2.5)", class = "sd")
)
if (mode != "known") priors <- c(priors, brms::set_prior("lkj(2)", class = "cor"))
dir.create(args[3], recursive = TRUE)
saveRDS(list(input = input, data = data, data2 = data2, settings = settings,
             phase = phase, mode = mode, identity = identity), file.path(args[3], "input.rds"))
writeLines(capture.output(sessionInfo()), file.path(args[3], "session.txt"))
writeLines(brms::make_stancode(formula, data = data, data2 = data2,
                              family = brms::binomial(), prior = priors),
           file.path(args[3], "generated.stan"))
saveRDS(brms::make_standata(formula, data = data, data2 = data2,
                          family = brms::binomial(), prior = priors),
        file.path(args[3], "standata.rds"))
fit <- brms::brm(
  formula, data = data, data2 = data2, family = brms::binomial(), prior = priors,
  backend = "cmdstanr", chains = settings["chains"], cores = 1,
  iter = settings["warmup"] + settings["draws"], warmup = settings["warmup"],
  seed = settings["seed"], sample_prior = if (phase == "prior") "only" else "no",
  control = list(adapt_delta = 0.95),
  output_dir = normalizePath(args[3])
)
saveRDS(fit, file.path(args[3], "fit.rds"))
saveRDS(posterior::as_draws_array(fit), file.path(args[3], "draws.rds"))
writeLines("Sampling output saved. Diagnostics, predictive checks, and cross-engine comparison remain pending.",
           file.path(args[3], "STATUS.txt"))
