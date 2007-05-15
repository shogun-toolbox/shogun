#library(sg)
dyn.load('../../src/sg.so')
sg <- function(...) .External("sg",...,PACKAGE="sg")

eps <- 1e-7

test_kernels <- function(filename){
	source("read_mfile.R")
 	code_lines <- read_mfile(filename)
	print(ls())
	#res <- test_gaussian_kernel()
	eval(parse(text=paste('res <- ',functionname,"();")))
	return(res)
}



test_gaussian_kernel <- function(){

  sg('send_command', 'clean_features TRAIN');
  sg('set_features', 'TRAIN', traindat);

  kname <- paste("set_kernel GAUSSIAN REAL ", size_,width_)
  print(kname)
  sg('send_command', kname);

  sg('send_command', 'init_kernel TRAIN');
  trainkm <- sg('get_kernel_matrix');

  sg('set_features', 'TEST', testdat);
  sg('send_command', 'init_kernel TEST');
  testkm <- sg('get_kernel_matrix');
  print(traindat)
  print(testkm)
  print(paste('max testkm: ',max(max(testkm))))
  print(paste('max km_test: ',max(max(km_test))))
  print(paste('dim(testkm): ',dim(testkm)))
  print(paste('dim(km_test): ',dim(km_test)))
  #print(paste('dim(testkm): ',str(testkm), ' dim(km_test): ', dim(km_test)))
  a <- max(max(abs(km_test-testkm)))
  b <- max(max(abs(km_train-trainkm)))
  print(paste('a: ', a,' b: ', b)) 
  if(a+b<eps){
    result <- 0
  }
  else{
    result <- 1
  }
  return(result) 
}
test_chi2_kernel <- function(){return(1)}
test_linear_kernel<- function(){return(1)} 
test_poly_kernel<- function(){return(1)} 
test_sigmoid_kernel<- function(){return(1)} 
test_wdps_kernel<- function(){return(1)} 
