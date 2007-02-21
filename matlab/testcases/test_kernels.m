
function y = test_kernels(filename)
  addpath /.amd_mnt/huangho/export/kwaid0/home/jonas/shogun/trunk/src;
  addpath /.amd_mnt/huangho/export/kwaid0/home/jonas/shogun/trunk/python/testcases/mfiles;
  
  if (filename(end-1:end)=='.m')
     filename = filename(1:end-2)
     eval(filename);
     y = feval(functionname,filename);
     
  else
     fprintf(1,'File is not m-file')
  end

