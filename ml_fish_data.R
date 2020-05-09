
getwd()
fish = read.csv('fish_participant.csv', TRUE, ",")
class(fish)
head(fish)


plot(
  fish$Weight,
  fish$Height,
  col="blue",
  main="Weight vs Height",
  xlab="Weight",
  ylab="MPG",
  pch=20
)

  
  # fit the univariate model
  lm_univariate = lm(Weight ~ Height, data=fish)
  
  # fit a linear model of horsepower and weight on mpg
  lm_multivariate = lm(Weight ~ Height + Length2, data=fish)
  
  # see the details of the model
  summary(lm_multivariate)

  
  # do an F-test comparing the univariate model to the multivariate one
  anova(lm_multivariate, lm_univariate)
  
  # show the diagnostic plots
  par(mfrow=c(2, 2))
  plot(lm_univariate)