C <- 1000;
dims <- 2;
num <- 50;

require(graphics)
require(lattice)
library("sg")

newplot <- get(getOption('device'))

meshgrid <- function(a,b) {
  list(
       x=outer(b*0,a,FUN="+"),
       y=outer(b,a*0,FUN="+")
      )
}

#set.seed(17)

trySVM <- function(c, kernel="POLY REAL 50 3 1", wireframe=FALSE) {

  sg("set_features", "TRAIN", traindat)
  sg("set_labels", "TRAIN", trainlab)
  sg("send_command", paste("set_kernel", kernel))
  sg("send_command", "init_kernel TRAIN")
  sg("send_command", "new_svm LIBSVM")
  sg("send_command", paste("c", c))
  
  sg("send_command", "svm_train")
  
  x1 <- (-49:+50)/10
  x2 <- (-49:+50)/10
  testdat <- meshgrid(x1,x2)

  testdat <- t(matrix(c(testdat$x, testdat$y),10000,2))
  
  sg("set_features", "TEST", testdat)
  sg("send_command", "init_kernel TEST")
  out <- sg("svm_classify")
  
  z <- t(matrix(out, 100, 100))
  
  svs <- sg("get_svm")$SV+1
  
  if (wireframe == TRUE)
  {
#for some reason plots only when done interactively
	  wireframe(z, shade = TRUE, aspect = c(61/87, 0.4), light.source = c(10,0,10))
  }
 else
 {
    image(x1,x2,z,col=topo.colors(1000))
    contour(x1,x2,z,add=T)
    cat(length(svs), "svs:", svs, "\n")

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
trySVM(C)
newplot()
trySVM(C, kernel="LINEAR REAL 100 1.0")
newplot()
trySVM(C, kernel="GAUSSIAN REAL 40 1.0")
