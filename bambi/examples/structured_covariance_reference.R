library(glmmTMB)
library(jsonlite)

output <- commandArgs(trailingOnly=TRUE)[1]
dir.create(output, recursive=TRUE, showWarnings=FALSE)
data <- data.frame(
    y=c(0.4, -0.5, 0.7, 0.1, -0.2, 0.8, NA),
    x=c(-1, 0.2, 0.8, 1.3, -0.6, 0.5, 0),
    time=c(0, 1, 1, 3, 4, 0, 2),
    g=factor(rep("a", 7)),
    visit=factor(c("a", "b", "a", "c", "b", "c", "a"))
)
data$grid <- factor(data$time, levels=0:4)
data$elapsed <- numFactor(data$time)
sigma <- 0.3
beta <- c(0.2, 0.4)
rho <- -0.35
fixed_optimizer <- function(par, fn, gr, control, ...) {
    list(par=par, objective=fn(par), convergence=0L,
         message="All population parameters fixed", iterations=0L,
         evaluations=c(functions=1L, gradient=0L))
}

records <- list()
for (structure in c("ar1", "ou", "diag", "cs", "cs_homogeneous", "us")) {
    if (structure == "ar1") {
        expression <- "ar1(0+grid|g)"
        theta <- c(log(0.7), rho/sqrt(1-rho^2))
    } else if (structure == "ou") {
        expression <- "ou(0+elapsed|g)"
        theta <- c(log(0.7), log(0.4))
    } else if (structure %in% c("cs", "cs_homogeneous")) {
        expression <- "cs(0+visit|g)"
        scales <- if (structure == "cs") c(0.7, 0.5, 0.4) else rep(0.7, 3)
        theta <- c(log(scales), qlogis((rho+0.5)/1.5))
    } else {
        expression <- paste0(structure, "(1+x|g)")
        theta <- log(c(0.7, 0.4))
        if (structure == "us") theta <- c(theta, rho/sqrt(1-rho^2))
    }
    fit <- glmmTMB(
        as.formula(paste("y ~ 1+x+", expression)), data=data,
        start=list(beta=beta, theta=theta, betadisp=log(sigma)),
        map=list(beta=factor(rep(NA, 2)), theta=factor(rep(NA, length(theta))),
                 betadisp=factor(NA)),
        se=FALSE, na.action=na.pass, control=glmmTMBControl(optimizer=fixed_optimizer)
    )
    fit$obj$fn(fit$fit$par)
    conditional_covariance <- solve(as.matrix(fit$obj$env$spHess(random=TRUE)))
    conditional_mode <- fit$obj$env$last.par.best[fit$obj$env$random]
    observed <- !is.na(data$y)
    records[[structure]] <- list(
        covariance=unname(as.matrix(VarCorr(fit)$cond$g)),
        design=unname(as.matrix(getME(fit, "Z"))[observed, ]),
        mean=as.vector(model.matrix(fit)[observed, ] %*% beta),
        y=data$y[observed], time=data$time[observed], sigma=as.numeric(sigma(fit)),
        conditional_mean=as.vector(conditional_mode),
        conditional_covariance=unname(conditional_covariance),
        predicted_mean=as.vector(predict(fit, type="response"))[observed],
        log_likelihood=-fit$obj$fn(fit$fit$par),
        theta=theta
    )
}
write_json(list(R=R.version.string, glmmTMB=as.character(packageVersion("glmmTMB")),
                cases=records), file.path(output, "glmmTMB.json"),
           auto_unbox=TRUE, digits=16, pretty=TRUE)
capture.output(sessionInfo(), file=file.path(output, "R-session.txt"))
