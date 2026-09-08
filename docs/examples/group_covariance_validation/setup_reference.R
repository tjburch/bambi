# Install the isolated CI reference toolchain, never the user's local environment.
if (Sys.getenv("GITHUB_ACTIONS") != "true") stop("This installer is restricted to GitHub Actions")
if (!nzchar(Sys.getenv("GITHUB_PAT"))) stop("GITHUB_PAT must contain the CI job token")
Sys.setenv(MAKEFLAGS = "-j1", STAN_NUM_THREADS = "1")
# Keep the binary repository and HTTP user agent configured by setup-r.
options(Ncpus = 1)
if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
has_version <- function(package, version) {
  tryCatch(as.character(packageVersion(package)) == version, error = function(e) FALSE)
}
# RStan is a brms import even when fits use CmdStanR. Keep its headers compatible.
versions <- c(posterior = "1.6.1", jsonlite = "2.0.0", StanHeaders = "2.32.10",
              rstan = "2.32.7", brms = "2.23.0")
for (package in names(versions)) {
  if (!has_version(package, versions[[package]])) {
    remotes::install_version(package, version = versions[[package]],
                             dependencies = NA, upgrade = "never")
  }
  if (!has_version(package, versions[[package]])) {
    stop("Installation failed for ", package, " ", versions[[package]])
  }
}
if (!has_version("cmdstanr", "0.9.0")) {
  remotes::install_github("stan-dev/cmdstanr@v0.9.0", dependencies = NA, upgrade = "never")
}
for (package in names(c(versions, cmdstanr = "0.9.0"))) {
  version <- c(versions, cmdstanr = "0.9.0")[[package]]
  if (!has_version(package, version) || !requireNamespace(package, quietly = TRUE)) {
    stop("Reference package is missing, unloadable, or has the wrong version: ", package)
  }
}
destination <- file.path(Sys.getenv("RUNNER_TEMP"), "covariance-cmdstan")
dir.create(destination, recursive = TRUE, showWarnings = FALSE)
version <- "2.37.0"
path <- file.path(destination, paste0("cmdstan-", version))
if (!dir.exists(path)) cmdstanr::install_cmdstan(dir = destination, version = version, cores = 1)
cmdstanr::set_cmdstan_path(path)
cat(paste0("CMDSTAN=", path, "\n"), file = Sys.getenv("GITHUB_ENV"), append = TRUE)
write.csv(installed.packages()[, c("Package", "Version")], "r-reference-packages.csv", row.names = FALSE)
