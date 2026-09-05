library(multcomp)
data(recovery)

means <- tapply(recovery$minutes, recovery$blanket, mean)
n <- table(recovery$blanket)

fit <- aov(minutes ~ blanket, data = recovery)
sigma2 <- summary(fit)[[1]]["Residuals", "Mean Sq"]

z <- c(
  means["b1"] - means["b0"],
  means["b2"] - means["b0"],
  means["b3"] - means["b0"]
)

Sigma <- sigma2 * matrix(c(
  1/n["b1"] + 1/n["b0"], 1/n["b0"],             1/n["b0"],
  1/n["b0"],             1/n["b2"] + 1/n["b0"], 1/n["b0"],
  1/n["b0"],             1/n["b0"],             1/n["b3"] + 1/n["b0"]
), 3, 3, byrow = TRUE)

z.stat <- z / sqrt(diag(Sigma))
p.val <- pnorm(z.stat)

z
Sigma
z.stat
p.val
