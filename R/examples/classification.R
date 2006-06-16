require(graphics)
require(lattice)
library("sg")

meshgrid <- function(a,b) {
	list(
			x=outer(b*0,a,FUN="+"),
			y=outer(b,a*0,FUN="+")
		)
} 

dims=2;
num=100;

traindat <- matrix(c(rnorm(dims*num)-0.5,rnorm(dims*num)+0.5),dims,2*num)
trainlab <- c(vector(mode="numeric", num)-1, vector(mode="numeric", num)+1)

sg("send_command","loglevel ALL")
sg("set_features", "TRAIN", traindat)
sg("set_labels", "TRAIN", trainlab)
sg("send_command", "set_kernel GAUSSIAN REAL 40 1")
sg("send_command", "init_kernel TRAIN")
sg("send_command", "new_svm LIGHT")
sg("send_command", "c 10.0")
sg("send_command", "svm_train")

x1=(-49:+50)/10
x2=(-49:+50)/10
testdat=meshgrid(x1,x2)
testdat=t(matrix(c(testdat$x, testdat$y),10000,2))

sg("set_features", "TEST", testdat)
sg("send_command", "init_kernel TEST")
out=sg("svm_classify")

z=matrix(out, 100,100)

image(x1,x2,z,col=topo.colors(1000))
contour(x1,x2,z,add=T)
i=which(trainlab==+1);
matplot(traindat[1,i],traindat[2,i],cex=2.0,pch="o", type = "p", col="red",add=T)
i=which(trainlab==-1);
matplot(traindat[1,i],traindat[2,i],cex=2.0,pch="x", type = "p", col="black",add=T)

wireframe(z, shade = TRUE, aspect = c(61/87, 0.4), light.source = c(10,0,10))
