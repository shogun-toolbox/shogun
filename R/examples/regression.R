require(graphics)

traindat <- array(1:1000)/10
trainlab <- sin(traindat)
testdat <- (array(1:1000)-1)/10
testlab <- testdat

library("sg")
sg("send_command","loglevel ALL")
sg("set_features", "TRAIN", traindat)
sg("set_labels", "TRAIN", trainlab)
sg("send_command", "set_kernel GAUSSIAN REAL 50 20")
sg("send_command", "init_kernel TRAIN")
sg("send_command", "new_svm LIBSVR")
sg("send_command", "c 1.0")
sg("send_command", "svm_train")
sg("set_features", "TEST", testdat)
sg("set_labels", "TEST", testlab)
sg("send_command", "init_kernel TEST")
out <- sg("svm_classify")
plot(testdat,out,type='o')
