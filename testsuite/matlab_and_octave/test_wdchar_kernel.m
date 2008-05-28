function y = test_wdchar_kernel(filename)

  eval(filename);

  sg('set_features', 'TRAIN', traindat, 'DNA');

  sg('send_command',sprintf('set_kernel WEIGHTEDDEGREE CHAR %i',degree));
  sg('send_command', 'init_kernel TRAIN');
  trainkm = sg('get_kernel_matrix');

  sg('set_features', 'TEST', testdat, 'DNA');
  sg('send_command', 'init_kernel TEST');
  testkm = sg('get_kernel_matrix');
  
  

  orgkm = km_test;
  newkm = testkm;
  

  a = max(max(abs(km_test-testkm)));
  b = max(max(abs(km_train-trainkm)));
  if(a+b<1e-7)
    y = 0;
  else
    y = 1;
  end