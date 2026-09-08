# Export common posterior mean metrics without hiding missing sampler diagnostics.
Sys.setenv(OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
           VECLIB_MAXIMUM_THREADS = "1")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5L) {
  stop("Usage: export_summary.R brms|stan mode fit.rds fixture-data.csv output.json")
}
engine <- match.arg(args[1], c("brms", "stan"))
mode <- match.arg(args[2], c("ar1", "known", "us-slopes", "us-visits"))
if (file.exists(args[5])) stop("Output already exists")
if (!requireNamespace("posterior", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("posterior and jsonlite are required")
}
fit <- readRDS(args[3])
data <- read.csv(args[4])
identity <- jsonlite::read_json(file.path(dirname(args[4]), paste0("identity-", mode, ".json")),
                               simplifyVector = FALSE)
if (!identical(identity$data_md5, unname(tools::md5sum(args[4])))) stop("Fixture identity mismatch")
run_dir <- dirname(args[3])
if (engine == "brms") {
  if (!requireNamespace("brms", quietly = TRUE)) stop("brms is required")
  settings <- readRDS(file.path(run_dir, "input.rds"))
  if (settings$phase != "posterior" || settings$mode != mode) stop("Run phase or mode mismatch")
  for (name in c("y", "trials", "x1", "x2", "subject", "condition", "context", "year")) {
    if (!identical(as.character(fit$data[[name]]), as.character(data[[name]]))) {
      stop("Fixture data do not match the saved fit")
    }
  }
  sampler <- fit$fit
  raw <- posterior::as_draws_array(fit)
} else {
  if (mode != "ar1") stop("The independent Stan runner currently supports AR1 only")
  settings <- readRDS(file.path(run_dir, "settings.rds"))
  if (settings$phase != "posterior") stop("Only posterior fits can be compared")
  input <- jsonlite::read_json(file.path(run_dir, "data.json"), simplifyVector = TRUE)
  if (!isTRUE(all.equal(as.numeric(input$y), as.numeric(data$y), tolerance = 0)) ||
      !isTRUE(all.equal(as.numeric(input$trials), as.numeric(data$trials), tolerance = 0)) ||
      !isTRUE(all.equal(unname(input$X), unname(as.matrix(cbind(1, data$x1, data$x2))), tolerance = 0))) {
    stop("Fixture data do not match the saved fit")
  }
  sampler <- fit
  raw <- fit$draws(format = "draws_array")
  groups <- list("subject", c("subject", "condition"), c("subject", "context"),
                 c("subject", "condition", "context"))
  for (b in seq_len(4L)) {
    indices <- input$row_cell[, b]
    if (!all(input$time[indices] == data$year) || !all(input$block_id[indices] == b)) {
      stop("Stan time or block mapping differs from fixture")
    }
    key <- do.call(paste, c(data[groups[[b]]], sep = ":"))
    mapping <- unique(data.frame(key = key, group = input$group_id[indices]))
    if (anyDuplicated(mapping$key) || anyDuplicated(mapping$group)) {
      stop("Stan group mapping differs from fixture")
    }
  }
}
if (!identical(settings$identity, identity)) stop("Saved fit identity differs from fixture identity")
if (!identical(as.numeric(unlist(identity$design$row_times)), as.numeric(data$year)) ||
    !identical(as.numeric(unlist(identity$design$trials)), as.numeric(data$trials)) ||
    !identical(as.numeric(unlist(identity$design$response)), as.numeric(data$y))) {
  stop("Identity observation design differs from fixture")
}
iterations <- dim(raw)[1]
chains <- dim(raw)[2]
variables <- dimnames(raw)[[3]]
metrics <- list()
add_metric <- function(name, source) {
  if (!source %in% variables) stop(paste("Missing required posterior variable:", source))
  metrics[[name]] <<- raw[, , source]
}
for (i in seq_len(3L)) {
  name <- c("one", "x1", "x2")[i]
  source <- if (engine == "brms") paste0("b_", name) else paste0("beta[", i, "]")
  add_metric(paste0("beta.", name), source)
}
if (engine == "stan") {
  for (n in seq_len(nrow(data))) add_metric(paste0("probability.", n), paste0("probability[", n, "]"))
  for (n in seq_len(nrow(data))) add_metric(paste0("log_likelihood.", n), paste0("log_lik[", n, "]"))
} else {
  probability <- brms::posterior_linpred(fit, transform = TRUE, re_formula = NULL)
  if (length(dim(probability)) != 2L || any(dim(probability) != c(iterations * chains, nrow(data)))) {
    stop("Prediction dimensions differ")
  }
  for (n in seq_len(nrow(data))) metrics[[paste0("probability.", n)]] <- matrix(probability[, n], iterations, chains)
  likelihood <- brms::log_lik(fit)
  if (!identical(dim(likelihood), dim(probability))) stop("Log likelihood dimensions differ")
  for (n in seq_len(nrow(data))) metrics[[paste0("log_likelihood.", n)]] <- matrix(likelihood[, n], iterations, chains)
}
for (n in seq_len(nrow(data))) {
  p <- metrics[[paste0("probability.", n)]]
  expectation <- data$trials[n] * p
  metrics[[paste0("predictive_mean.", n)]] <- expectation
  metrics[[paste0("predictive_second_moment.", n)]] <- data$trials[n] * p * (1 - p) + expectation^2
  metrics[[paste0("predictive_zero_probability.", n)]] <- (1 - p)^data$trials[n]
}
if (mode %in% c("ar1", "known")) {
  for (b in seq_len(4L)) {
    source <- if (engine == "stan") paste0("sd[", b, "]") else paste0("sd_cell", b, "__Intercept")
    add_metric(paste0("sd.", b), source)
    if (mode == "ar1") add_metric(paste0("rho.", b), paste0("rho[", b, "]"))
    for (cell in unique(data[[paste0("cell", b)]])) {
      rows <- which(data[[paste0("cell", b)]] == cell)
      if (engine == "stan") {
        indices <- unique(input$row_cell[rows, b])
        if (length(indices) != 1L) stop("Cell maps to multiple Stan coefficients")
        source <- paste0("coefficient[", indices, "]")
      } else {
        source <- paste0("r_cell", b, "[", cell, ",Intercept]")
      }
      add_metric(paste0("latent.", cell), source)
    }
  }
} else {
  coefficients <- if (mode == "us-slopes") c("Intercept", "x1") else paste0("visit", levels(fit$data$visit))
  for (i in seq_along(coefficients)) add_metric(paste0("sd.", i), paste0("sd_subject__", coefficients[i]))
  for (subject in unique(data$subject)) {
    for (i in seq_along(coefficients)) {
      add_metric(paste0("latent.subject.", subject, ".", i),
                 paste0("r_subject[", subject, ",", coefficients[i], "]"))
    }
  }
  for (i in seq_along(coefficients)) {
    if (i < length(coefficients)) {
      for (j in seq.int(i + 1L, length(coefficients))) {
        add_metric(paste0("cor.", i, ".", j), paste0("cor_subject__", coefficients[i], "__", coefficients[j]))
      }
    }
  }
}
metric_array <- array(unlist(metrics), c(iterations, chains, length(metrics)),
                      dimnames = list(NULL, NULL, names(metrics)))
summarize <- function(draws) posterior::summarise_draws(
  posterior::as_draws_array(draws), mean = mean, mcse_mean = posterior::mcse_mean,
  rhat = posterior::rhat, ess_bulk = posterior::ess_bulk, ess_tail = posterior::ess_tail
)
summary <- summarize(metric_array)
constant <- vapply(metrics, function(values) all(values == values[1]), logical(1L))
summary$mcse_mean[constant[summary$variable]] <- 0
pattern <- if (engine == "stan") "^(beta\\[|sd\\[|rho\\[|z\\[)" else "^(b_|sd_|cor_|r_)"
parameter_names <- variables[grepl(pattern, variables)]
if (!length(parameter_names)) stop("Cannot locate sampled parameters for diagnostics")
parameter_summary <- summarize(raw[, , parameter_names, drop = FALSE])
deterministic <- grepl("^(probability\\.|log_likelihood\\.|predictive_)", summary$variable) &
  constant[summary$variable]
diagnostic_table <- rbind(summary[!deterministic, ], parameter_summary)
stats <- sampler$sampler_diagnostics(format = "draws_array")
required <- c("divergent__", "energy__", "treedepth__")
if (!all(required %in% dimnames(stats)[[3]])) stop("Missing sampler diagnostics")
max_depth <- sampler$metadata()$max_treedepth
if (is.null(max_depth) || anyNA(max_depth)) stop("Missing maximum tree depth setting")
energy <- stats[, , "energy__"]
bfmi <- vapply(seq_len(chains), function(chain) mean(diff(energy[, chain])^2) / var(energy[, chain]), numeric(1L))
diagnostics <- list(
  chains = chains, rhat_max = max(diagnostic_table$rhat),
  ess_bulk_min = min(diagnostic_table$ess_bulk), ess_tail_min = min(diagnostic_table$ess_tail),
  divergences = sum(stats[, , "divergent__"]), bfmi_min = min(bfmi),
  treedepth_hits = sum(sweep(stats[, , "treedepth__"], 2L, rep(max_depth, length.out = chains), ">="))
)
if (any(!is.finite(unlist(diagnostics))) || any(!is.finite(summary$mean)) ||
    any(!is.finite(summary$mcse_mean))) stop("Nonfinite metrics or diagnostics; no certified summary written")
exported <- setNames(lapply(seq_len(nrow(summary)), function(i) {
  values <- metrics[[summary$variable[i]]]
  probabilities <- c(0.03, 0.5, 0.97)
  quantiles <- setNames(lapply(probabilities, function(p) {
    list(value = unname(quantile(values, probs = p)),
         mcse = if (constant[summary$variable[i]]) 0 else unname(posterior::mcse_quantile(values, probs = p)))
  }), c("0.03", "0.5", "0.97"))
  if (any(!is.finite(unlist(quantiles)))) stop("Nonfinite quantile summary")
  list(mean = summary$mean[i], mcse_mean = summary$mcse_mean[i], quantiles = quantiles)
}), summary$variable)
jsonlite::write_json(
  list(schema_version = 2L, identity = identity, engine = engine, mode = mode, phase = "posterior",
       fixed_rho = if (mode == "known") settings$input$fixed_rho else NULL,
       data_md5 = unname(tools::md5sum(args[4])), diagnostics = diagnostics, metrics = exported,
       coverage = "fixed effects, latent effects, covariance, probability, log likelihood, predictive moments"),
  args[5], auto_unbox = TRUE, pretty = TRUE, digits = NA
)
