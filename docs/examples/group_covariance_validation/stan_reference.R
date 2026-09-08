# Serial runner for the independent four-block AR1 model.
Sys.setenv(
  OMP_NUM_THREADS = "1", OPENBLAS_NUM_THREADS = "1", MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1", STAN_NUM_THREADS = "1", MAKEFLAGS = "-j1"
)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop("Usage: stan_reference.R model.stan data.json output-dir prior|posterior chains warmup draws seed")
}
phase <- match.arg(args[4], c("prior", "posterior"))
settings <- suppressWarnings(as.integer(args[5:8]))
if (anyNA(settings) || any(settings < 1L)) stop("Sampling settings must be positive integers")
names(settings) <- c("chains", "warmup", "draws", "seed")
if (!requireNamespace("cmdstanr", quietly = TRUE) || !requireNamespace("jsonlite", quietly = TRUE)) {
  stop("Install cmdstanr, jsonlite, and CmdStan in a separate reference environment first")
}
if (file.exists(args[3])) stop("Output path already exists; use a separate run directory")
data <- jsonlite::read_json(args[2], simplifyVector = TRUE)
identity <- jsonlite::read_json(file.path(dirname(args[2]), "identity-ar1.json"),
                               simplifyVector = FALSE)
if (!identical(identity$data_md5, unname(tools::md5sum(file.path(dirname(args[2]), "data.csv"))))) {
  stop("Fixture identity mismatch")
}
data$prior_only <- as.integer(phase == "prior")
dir.create(args[3], recursive = TRUE)
file.copy(args[1], file.path(args[3], "model.stan"))
jsonlite::write_json(data, file.path(args[3], "data.json"), auto_unbox = TRUE, digits = 17)
saveRDS(list(settings = settings, phase = phase, identity = identity), file.path(args[3], "settings.rds"))
writeLines(capture.output(sessionInfo()), file.path(args[3], "session.txt"))
model <- cmdstanr::cmdstan_model(file.path(args[3], "model.stan"), quiet = FALSE)
fit <- model$sample(
  data = data, chains = settings["chains"], parallel_chains = 1,
  threads_per_chain = 1, iter_warmup = settings["warmup"],
  iter_sampling = settings["draws"], seed = settings["seed"],
  adapt_delta = 0.95, output_dir = normalizePath(args[3])
)
fit$save_object(file.path(args[3], "fit.rds"))
writeLines("Sampling output saved. Diagnostics, predictive checks, and cross-engine comparison remain pending.",
           file.path(args[3], "STATUS.txt"))
