library(sg)

acgt <- c("A","C","G","T") 
XT= array("",dim=c(100,1000))

for (i in 1:length(XT)) {
   XT[i] = acgt[ceiling(4 * (rnorm(1) %% 1))]
}

LT=sign(rnorm(1000)) 

for (k in c(30,60,61)) {
   for (i in 1:length(XT[k,])) {
      if (LT[i] == 1) {
         XT[k,i] = "A"
      }
   }
}

idx=sample(c(1:1000)) 
XTE=XT[,idx[1:200]] 
LTE=LT[idx[1:200]] 
XT=XT[,901:1000] 
LT=LT[901:length(LT)]

take_idx = c(1:100)
center_idx = 50
degree=3
mismatch = 0
C=1

send_command( "loglevel ALL") 
send_command("use_mkl 0") 
send_command("use_linadd 1") 
send_command("use_precompute 0") 
send_command( "mkl_parameters 1e-5 1") 
send_command( "svm_epsilon 1e-6") 
send_command( "clean_features TRAIN") 
send_command( "clean_kernels") 
send_command( "new_svm LIGHT") 

set_labels("TRAIN", LT)
set_features("TRAIN", XT[take_idx,])
send_command( sprintf("set_kernel WEIGHTEDDEGREE CHAR 10 %i %i 0 0", degree, mismatch)) 

send_command( "init_kernel TRAIN") 
send_command( sprintf("c %1.2e", C)) 
send_command( "svm_train") 

svmAsList=sg("get_svm") 
beta=sg("get_subkernel_weights") 

send_command( "init_kernel_optimization") 

send_command( "clean_features TEST") 
set_features("TEST", XTE[take_idx,]) 

send_command( "init_kernel TEST") 
output_xte = svm_classify()
