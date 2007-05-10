source('test_kernels.R')
test_R <- function(filename){
        print('test_R')
	#vec <- unlist(strsplit(filename,"/"))
  	#cat(vec[length(vec)])
	res <- test_kernels(filename);#vec[length(vec)])

 	if (res==0){
	print('__OK__')
	}
	else{
	print('__error__')
	}
	quit(save='no')
}


