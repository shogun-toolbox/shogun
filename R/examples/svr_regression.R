require(graphics)
library("sg")

traindat <- matrix(1:1000+runif(1000),1,1000)/10
trainlab <- sin(traindat)
testdat <- (matrix(1:1000,1,1000)-1+runif(1000))/10
testlab <- testdat

sg("send_command","loglevel ALL")
sg("set_features", "TRAIN", traindat)
sg("set_labels", "TRAIN", trainlab)
sg("send_command", "set_kernel GAUSSIAN REAL 50 20")
sg("send_command", "init_kernel TRAIN")
sg("send_command", "new_svm LIBSVR")
sg("send_command", "c 0.1")
sg("send_command", "svm_train")
sg("set_features", "TEST", testdat)
sg("set_labels", "TEST", testlab)
sg("send_command", "init_kernel TEST")
out <- sg("svm_classify")
plot(traindat,trainlab, type = "o", pch="x", col="red");
matplot(testdat,out, type = "o", pch="o", col="black",add=T)
