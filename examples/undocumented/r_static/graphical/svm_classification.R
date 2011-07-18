
C <- 1000;
dims <- 2;
num <- 50;

require(graphics)
#require(lattice)
library('sg')

#uncomment if make install does not work and comment the library("sg") line above
#dyn.load('sg.so')
#sg <- function(...) .External("sg",...,PACKAGE="sg")

#newplot <- get(getOption('device'))

meshgrid <- function(a,b) {
  list(
       x=outer(b*0,a,FUN="+"),
       y=outer(b,a*0,FUN="+")
      )
}

#set.seed(17)

trySVM <- function(c, ktype, dtype, size_cache, wireframe=FALSE) {

  sg('set_features', 'TRAIN', traindat)
  sg('set_labels', 'TRAIN', trainlab)
  sg('set_kernel', ktype, dtype, size_cache)
  sg('new_classifier', 'LIBSVM')
  sg('c', c)

  sg('train_classifier')

  x1 <- (-49:+50)/10
  x2 <- (-49:+50)/10
  testdat <- meshgrid(x1,x2)

  testdat <- t(matrix(c(testdat$x, testdat$y),10000,2))

  sg('set_features', 'TEST', testdat)
  out <- sg('classify')

  z <- t(matrix(out, 100, 100))

  svm <- sg('get_svm')
  b=svm[[1]]
  svs=svm[[2]][,2]+1

  objective <- sg('get_svm_objective')
  print(objective)

  if (wireframe == TRUE)
  {
#for some reason plots only when done interactively
	  wireframe(z, shade = TRUE, aspect = c(61/87, 0.4), light.source = c(10,0,10))
  }
 else
 {
    image(x1,x2,z,col=topo.colors(1000))
    contour(x1,x2,z,add=T)
    cat(length(svs), 'svs:', svs, "\n")

    posSVs <- traindat[,trainlab==+1 & (1:ncol(traindat) %in% svs)]
    negSVs <- traindat[,trainlab==-1 & (1:ncol(traindat) %in% svs)]

    pos    <- traindat[,trainlab==+1 & !(1:ncol(traindat) %in% svs)]
    neg    <- traindat[,trainlab==-1 & !(1:ncol(traindat) %in% svs)]

    matplot(posSVs[1,],posSVs[2,], pch="+", col="red",add=T, cex=1.5)
    matplot(negSVs[1,],negSVs[2,], pch="-", col="red",  add=T, cex=2.0)

    matplot(pos[1,],pos[2,], pch="+", col="black",add=T, cex=1.5)
    matplot(neg[1,],neg[2,], pch="-", col="black",add=T, cex=2.0)
 }

}

traindat <- matrix(c(rnorm(dims*num)-1,rnorm(dims*num)+1),dims,2*num)
trainlab <- c(rep(-1,num),rep(1,num))

graphics.off()
trySVM(C, 'SIGMOID', 'REAL', 50)
#newplot()
trySVM(C, 'LINEAR', 'REAL', 100)
#newplot()
trySVM(C, 'GAUSSIAN', 'REAL', 40)
