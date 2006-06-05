require(graphics)

traindat <- array(1:1000)/10
trainlab <- sin(traindat)
testdat <- (array(1:1000)-1)/10
testlab <- testdat

library("sg")
#send_command("loglevel ALL")
set_features("TRAIN", traindat)
set_labels("TRAIN", trainlab)
send_command("set_kernel GAUSSIAN REAL 50 20")
send_command("init_kernel TRAIN")
send_command("new_svm LIBSVR")
send_command("c 1.0")
send_command("svm_train")
set_features("TEST", testdat)
set_labels("TEST", testlab)
send_command("init_kernel TEST")
out <- svm_classify()
plot(testdat,out,type='o')
