#library(sg)
dyn.load('../sg/src/sg.so')
eps <- 1e-7
source("../sg/R/shogun.R")
test_kernels <- function(filename){

	source("read_mfile.R")
 	code_lines <- read_mfile(filename)
	tcon <- textConnection(paste('res <- ',functionname,"(code_lines)"))
	source(tcon)
	close(tcon)
	return(res)
}


test_gaussian_kernel <- function(code_lines){

  tcon <- textConnection(code_lines)
  source(tcon)
  close(tcon)

  send_command('clean_features TRAIN');
  set_features( 'TRAIN', traindat);

  kname <- paste("set_kernel GAUSSIAN REAL ", size_,width_)
  send_command(kname);

  send_command('init_kernel TRAIN');
  trainkm <- get_kernel_matrix();

  set_features( 'TEST', testdat);
  send_command('init_kernel TEST');
  testkm <- get_kernel_matrix();
  print(paste('max testkm: ',max(max(testkm))))
  print(paste('max km_test: ',max(max(km_test))))
  print(paste('dim(testkm): ',dim(testkm)))
  print(paste('dim(km_test): ',dim(km_test)))
  #print(paste('dim(testkm): ',str(testkm), ' dim(km_test): ', dim(km_test)))
  a <- max(max(abs(km_test-testkm)))
  b <- max(max(abs(km_train-trainkm)))
  print(paste('a: ', a,' b: ', b)) 
  if(a+b<1){
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
