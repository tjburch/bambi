# Install the isolated CI reference toolchain, never the user's local environment.
if (Sys.getenv("GITHUB_ACTIONS") != "true") stop("This installer is restricted to GitHub Actions")
Sys.setenv(MAKEFLAGS = "-j1", STAN_NUM_THREADS = "1")
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 1)
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
versions <- c(brms = "2.23.0", posterior = "1.6.1", jsonlite = "2.0.0")
for (package in names(versions)) {
  if (!requireNamespace(package, quietly = TRUE) ||
      as.character(packageVersion(package)) != versions[[package]]) {
    remotes::install_version(package, version = versions[[package]],
                             dependencies = NA, upgrade = "never")
  }
}
if (!requireNamespace("cmdstanr", quietly = TRUE) ||
    as.character(packageVersion("cmdstanr")) != "0.9.0") {
  remotes::install_github("stan-dev/cmdstanr@v0.9.0", dependencies = NA, upgrade = "never")
}
destination <- file.path(Sys.getenv("RUNNER_TEMP"), "covariance-cmdstan")
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
version <- "2.37.0"
path <- file.path(destination, paste0("cmdstan-", version))
if (!dir.exists(path)) cmdstanr::install_cmdstan(dir = destination, version = version, cores = 1)
cmdstanr::set_cmdstan_path(path)
cat(paste0("CMDSTAN=", path, "\n"), file = Sys.getenv("GITHUB_ENV"), append = TRUE)
write.csv(installed.packages()[, c("Package", "Version")], "r-reference-packages.csv", row.names = FALSE)
