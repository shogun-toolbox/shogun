# Blind Source Separation using the Jade Algorithm with Shogun
#
# Based on the example from scikit-learn
# http://scikit-learn.org/
#
# Kevin Hughes 2013

library('sg')

# Generate sample data
n_samples <- 2000
time <- seq(0,10,length=n_samples)

# Source Signals
S <- matrix(0,2,n_samples)
S[1,] <- sin(2*time)
S[2,] <- sign(sin(3*time))
S <- S + 0.2*matrix(runif(2*n_samples),2,n_samples)

# Standardize data
S <- S * (1/apply(S,1,sd))

# Mixing Matrix
A <- rbind(c(1,0.5),c(0.5,1))

# Mix Signals
X <- A %*% S
mixed_signals <- matrix(X,2,n_samples)

# Separating
sg('set_converter', 'jade')
sg('set_features', 'TRAIN', mixed_signals)
S_ <- sg('apply_converter')

# Plot
par(mfcol=c(3,1));

plot(time, S[1,], type="l", col='blue', main="True Sources", ylab="", xlab="")
lines(time, S[2,], type="l", col='green')

plot(time, X[1,], type="l", col='blue', main="Mixed Sources", ylab="", xlab="")
lines(time, X[2,], type="l", col='green')

plot(time, S_[1,], type="l", col='blue', main="Estimated Sources", ylab="", xlab="")
lines(time, S_[2,], type="l", col='green')
