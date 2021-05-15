### EGARCH modelling ###
for (i in 1:(N.per-1)) {
x <- array(0, dim = c(2,2,3,3))
for (p in 1:2) {for (q in 1:2) {for(a in 1:3) {for(b in 1:3) {
  tryCatch({
    x[p,q,a,b] <- as.numeric(head(infocriteria(ugarchfit(ugarchspec(variance.model = list(model = "eGARCH", garchOrder = c(p,q)), 
    mean.model = list(armaOrder = c(a,b)), distribution.model = "jsu"),
    coredata(returns[[i]]), solver = "hybrid")),n=1))
   }, 
   error=function(e){cat("ERROR :",conditionMessage(e), "\n")})}}}}

model <- ugarchspec(variance.model = list(model ="eGARCH", 
  garchOrder = c(which(x == min(x), arr.ind = T)[1], which(x ==min(x), arr.ind = T)[2])),
  mean.model = list(armaOrder = c(which(x == min(x), arr.ind = T)[3], which(x == min(x), arr.ind = T)[4]),
  include.mean = TRUE),
  distribution.model = "jsu")

fit[i] <- ugarchfit(spec=model,data=returns[[i]],solver = "hybrid")
forecast[i] <- ugarch
forecast(fit[[i]], n.ahead = 1)
}

### Dynamic delta hedging and trading ###
q=1000
for (i in 2:N.per) {
t1 <- new("Option", spot = as.numeric(price[[i]][1]), strike = as.numeric(price[[i]][1]), 
  TTE = as.numeric(TTE[[i]][1]), rf.rate = rf.rate, div = div,
  vol = as.numeric(vol[[i]][1])/100, right = right,quantity = 1)
t2 <- new("Option", spot = as.numeric(price[[i]][1]), strike = as.numeric(price[[i]][1]), 
  TTE = as.numeric(TTE[[i]][1]),
  rf.rate = rf.rate, div = div,
  vol = as.numeric(vol[[i]][1])/100, right = -right, quantity = 1)

portfolioValue.t0[i] <- q*(priceBS(t1) + priceBS(t2))
delta.mat[[i]] <- rep(0,length(price[[i]]))
}
for (i in 2:N.per) {
  for (j in 1:length(price[[i]])){
    t1 <- new("Option", spot = as.numeric(price[[i]][j]), strike = as.numeric(price[[i]][1]),
      TTE = as.numeric(TTE[[i]][j]),
      rf.rate = rf.rate, div = div, vol = as.numeric(vol[[i]][j])/100,
      right = right, quantity = 1)
     t2 <- new("Option", spot = as.numeric(price[[i]][j]), strike = as.numeric(price[[i]][1]),
      TTE = as.numeric(TTE[[i]][j]), rf.rate = rf.rate, div = div,
      vol = as.numeric(vol[[i]][j])/100, right = -right, quantity = 1)
      delta.mat[[i]][j] <- round(q*(deltaBS(t1) + deltaBS(t2)))}}

for (i in 2:N.per) {
  Q.mat[[i]] <- c(0,-delta.mat[[i]])
  P.mat[[i]] <- c(as.numeric(price[[i]][1]),as.numeric(price[[i]]), as.numeric(price[[i]][length(price[[i]])]))}
for (i in 2:N.per){
  pl.mat <- PL.fast(Q.mat[[i]], P.mat[[i]])
  pl.vec[i] <- sum(pl.mat[1,])
if (as.logical(forecast[[i-1]]@forecast$sigmaFor*sqrt(252)*100 > 
  data[which(data$report_date==as.Date(unlist(attributes(forecast[[i-1]]@forecast$sigmaFor))[4])),15])) {
  pl.derivative[i] <- q*pmax(as.numeric(price[[i]][length(price[[i]])]) - as.numeric(price[[i]][1]),0) + 
  q*pmax(as.numeric(-price[[i]][length(price[[i]])])+as.numeric(price[[i]][1]),0) - portfolioValue.t0[i]
}
else {
  pl.derivative[i] <- -(q*pmax(as.numeric(price[[i]][length(price[[i]])]) - as.numeric(price[[i]][1]),0) +
                        q*pmax(as.numeric(-price[[i]][length(price[[i]])])+as.numeric(price[[i]][1]),0) - portfolioValue.t0[i])
}
net.pl[i] <- pl.vec[i] + pl.derivative[i]
}
