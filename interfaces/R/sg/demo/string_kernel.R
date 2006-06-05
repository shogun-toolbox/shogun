library(sg)

traindat = c("AAAAA", "CCCCC", "GGGGG", "TTTTT")
trainlab <- c(1,-1,-1,1,-1) 

#testdat <- array("",dim=c(5,3))
#testdat[1,] = c("T","C","A")
#testdat[2,] = c("T","C","A")
#testdat[3,] = c("T","C","A")
#testdat[4,] = c("T","C","A")
#testdat[5,] = c("T","C","A")

testdat = traindat

order = 2 
C = 1.0 

send_command("loglevel ALL") 
send_command("use_mkl 0") 
send_command("use_linadd 1") 
send_command("use_precompute 0") 
send_command("mkl_parameters 1e-5 0") 
send_command("svm_epsilon 1e-4") 
send_command("clean_features TRAIN") 
send_command("clean_kernels") 
set_features("TRAIN", traindat) 
#send_command(sprintf("convert TRAIN SIMPLE CHAR STRING CHAR DNA %i %i",order,order-1)) 
#send_command(sprintf("convert TRAIN STRING CHAR STRING WORD DNA %i %i",order,order-1)) 
send_command(sprintf("convert TRAIN SIMPLE CHAR SIMPLE WORD DNA %i %i",order,order-1)) 
send_command("add_preproc SORTWORD") 
send_command("attach_preproc TRAIN") 
set_labels("TRAIN", trainlab) 
send_command("new_svm LIGHT") 
send_command("set_kernel COMM WORD 10 1 FULL")
send_command(sprintf("c %1.2e", C)) 
send_command("init_kernel TRAIN") 
km=get_kernel_matrix()
send_command("svm_train") 
svmAsList=get_svm() 

set_features("TEST", testdat) 
send_command(sprintf("convert TEST SIMPLE CHAR STRING CHAR DNA %i %i",order,order-1)) 
send_command(sprintf("convert TEST STRING CHAR STRING WORD DNA %i %i",order,order-1)) 
#send_command("add_preproc SORTWORDSTRING") 
send_command("attach_preproc TEST") 
send_command("init_kernel_optimization") 
send_command("init_kernel TEST") 
valout=svm_classify()

