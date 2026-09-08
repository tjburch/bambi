# Serial posterior reference for all five covariance structures.
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
           VECLIB_MAXIMUM_THREADS = "1", STAN_NUM_THREADS = "1", MAKEFLAGS = "-j1")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 6L) stop("Usage: single_block_stan.R fixture-dir output-dir chains warmup draws seed")
settings <- suppressWarnings(as.integer(args[3:6]))
if (anyNA(settings) || any(settings < 1L)) stop("Sampling settings must be positive")
for (package in c("cmdstanr", "posterior", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing package", package))
}
if (file.exists(args[2])) stop("Output directory already exists")
fixture <- jsonlite::read_json(file.path(args[1], "fixture.json"), simplifyVector = TRUE)
if (fixture$mode == "four-block-ar1") stop("Use stan_reference.R for four-block comparisons")
expected_priors <- list(beta = "Normal(0,1.5)", sd = "HalfNormal(2.5)",
                        rho = "TruncatedNormal(0,0.5;structure bounds)", decay = "Exponential(1)",
                        partial = "TruncatedNormal(0,0.5;-1,1)", correlation = "LKJ(2)")
if (!identical(fixture$priors, expected_priors) || fixture$schema_version != 1L ||
    !fixture$family %in% c("bernoulli", "binomial") || !isTRUE(fixture$prior_draw)) {
  stop("Unsupported fixture or prior contract")
}
identity <- jsonlite::read_json(file.path(args[1], paste0("identity-", fixture$mode, ".json")))
data_path <- file.path(args[1], "data.csv")
if (!identical(identity$data_md5, unname(tools::md5sum(data_path)))) stop("Fixture hash mismatch")
input <- jsonlite::read_json(file.path(args[1], "data.json"), simplifyVector = TRUE)
observations <- read.csv(data_path)
block <- jsonlite::read_json(file.path(args[1], "fixture.json"), simplifyVector = FALSE)$blocks[[1]]
for (name in c("times", "group_id", "level_id")) block[[name]] <- unlist(block[[name]])
same <- function(a, b) isTRUE(all.equal(unname(a), unname(b), tolerance = 0, check.attributes = FALSE))
if (!same(input$X, as.matrix(cbind(1, observations$x1, observations$x2))) ||
    !same(input$y, observations$y) || !same(input$trials, observations$trials) ||
    !same(input$group_id, block$group_id) || !same(input$level_id, block$level_id) ||
    !same(input$time, block$times) || !same(input$time_index, as.integer(block$times)) ||
    input$max_lag != block$max_lag || input$N != nrow(observations) ||
    input$P != 3L || input$Q != length(block$times) || input$G != length(block$groups) ||
    input$use_design != 0L ||
    input$structure != match(fixture$mode, c("ar1", "ou", "cs", "toep", "us"))) {
  stop("Stan input differs from the fixture contract")
}
if (input$prior_only != 0L) stop("Posterior reference requires prior_only=0")
if (input$use_design == 0L) input$Z <- matrix(numeric(), 0L, input$Q)
dir.create(args[2], recursive = TRUE)
script_arg <- commandArgs()[grepl("^--file=", commandArgs())][1]
source_dir <- dirname(sub("^--file=", "", script_arg))
source <- file.path(source_dir, "structured_block.stan")
model <- cmdstanr::cmdstan_model(source, quiet = FALSE)
jsonlite::write_json(list(identity = identity, settings = settings),
                     file.path(args[2], "settings.json"), auto_unbox = TRUE, pretty = TRUE)
fit <- model$sample(data = input, chains = settings[1], parallel_chains = 1,
                   threads_per_chain = 1, iter_warmup = settings[2], iter_sampling = settings[3],
                   seed = settings[4], adapt_delta = 0.95, output_dir = normalizePath(args[2]))
fit$save_object(file.path(args[2], "fit.rds"))
raw <- fit$draws(format = "draws_array")
metrics <- list()
add <- function(name, source) {
  if (!source %in% dimnames(raw)[[3]]) stop(paste("Missing parameter", source))
  metrics[[name]] <<- raw[, , source]
}
for (i in seq_len(3L)) add(paste0("beta.", c("one", "x1", "x2")[i]), paste0("beta[", i, "]"))
for (i in seq_len(if (fixture$mode == "us") input$Q else 1L)) add(paste0("sd.", i), paste0("sd[", i, "]"))
if (fixture$mode %in% c("ar1", "cs")) add("rho.1", "rho[1]")
if (fixture$mode == "ou") add("decay.1", "decay[1]")
if (fixture$mode == "toep") for (i in seq_len(input$max_lag)) add(paste0("partial.", i), paste0("partial[", i, "]"))
if (fixture$mode == "us") {
  for (i in seq_len(input$Q - 1L)) for (j in (i + 1L):input$Q) {
    add(paste0("cor.", i, ".", j), paste0("correlation[", i, ",", j, "]"))
  }
}
for (g in seq_len(input$G)) for (q in seq_len(input$Q)) {
  add(paste0("coefficient.", g, ".", q), paste0("coefficient[", g, ",", q, "]"))
}
for (n in seq_len(input$N)) {
  add(paste0("probability.", n), paste0("probability[", n, "]"))
  add(paste0("log_likelihood.", n), paste0("log_lik[", n, "]"))
  p <- metrics[[paste0("probability.", n)]]
  expectation <- input$trials[n] * p
  metrics[[paste0("predictive_mean.", n)]] <- expectation
  metrics[[paste0("predictive_second_moment.", n)]] <- input$trials[n] * p * (1 - p) + expectation^2
  metrics[[paste0("predictive_zero_probability.", n)]] <- (1 - p)^input$trials[n]
}
metric_array <- array(unlist(metrics), c(dim(raw)[1:2], length(metrics)),
                      dimnames = list(NULL, NULL, names(metrics)))
summarize <- function(draws) posterior::summarise_draws(
  posterior::as_draws_array(draws), mean = mean, mcse_mean = posterior::mcse_mean,
  rhat = posterior::rhat, ess_bulk = posterior::ess_bulk, ess_tail = posterior::ess_tail)
summary <- summarize(metric_array)
parameters <- grep("^(beta\\[|sd\\[|rho\\[|decay\\[|partial\\[|z\\[)", dimnames(raw)[[3]], value = TRUE)
diagnostic_table <- rbind(summary, summarize(raw[, , parameters, drop = FALSE]))
stats <- fit$sampler_diagnostics(format = "draws_array")
if (!all(c("divergent__", "energy__", "treedepth__") %in% dimnames(stats)[[3]])) stop("Missing diagnostics")
depth <- fit$metadata()$max_treedepth
if (is.null(depth) || anyNA(depth)) stop("Missing tree-depth setting")
energy <- stats[, , "energy__"]
bfmi <- vapply(seq_len(settings[1]), function(i) mean(diff(energy[, i])^2) / var(energy[, i]), numeric(1))
diagnostics <- list(chains = settings[1], rhat_max = max(diagnostic_table$rhat),
                    ess_bulk_min = min(diagnostic_table$ess_bulk), ess_tail_min = min(diagnostic_table$ess_tail),
                    divergences = sum(stats[, , "divergent__"]), bfmi_min = min(bfmi),
                    treedepth_hits = sum(sweep(stats[, , "treedepth__"], 2L, rep(depth, length.out = settings[1]), ">=")))
if (any(!is.finite(unlist(diagnostics)))) stop("Nonfinite diagnostics")
exported <- setNames(lapply(seq_len(nrow(summary)), function(i) {
  values <- metrics[[summary$variable[i]]]
  quantiles <- setNames(lapply(c(0.03, 0.5, 0.97), function(p) {
    list(value = unname(quantile(values, probs = p)), mcse = unname(posterior::mcse_quantile(values, probs = p)))
  }), c("0.03", "0.5", "0.97"))
  if (any(!is.finite(unlist(quantiles)))) stop("Nonfinite quantiles")
  list(mean = summary$mean[i], mcse_mean = summary$mcse_mean[i], quantiles = quantiles)
}), summary$variable)
if (any(!is.finite(unlist(lapply(exported, function(x) c(x$mean, x$mcse_mean)))))) stop("Nonfinite summaries")
jsonlite::write_json(list(schema_version = 2L, engine = "stan", mode = fixture$mode,
                         phase = "posterior", identity = identity, diagnostics = diagnostics,
                         data_md5 = unname(tools::md5sum(data_path)), metrics = exported),
                     file.path(args[2], "summary.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
