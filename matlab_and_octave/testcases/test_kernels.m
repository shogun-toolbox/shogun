function y = test_kernels(filename)
  addpath ('../../src');
  addpath ('../../testsuite/data/kernel');
  
  if (filename(end-1:end)=='.m')
     filename = filename(1:end-2)
     eval(filename);
     y = feval(name,filename);
     
  else
     fprintf(1,'File is not m-file')
  end

