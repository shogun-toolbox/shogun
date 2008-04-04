function y = test_linear_kernel(filename)
  eval(filename);

  kname = ['set_kernel ', 'LINEAR REAL 10 1.0'];

  sg('set_features', 'TRAIN', traindat);
  sg('send_command', kname);
  sg('send_command', 'init_kernel TRAIN');
  trainkm = sg('get_kernel_matrix')

  sg('set_features', 'TEST', testdat);
  sg('send_command', 'init_kernel TEST');
  testkm = sg('get_kernel_matrix')

  a = max(max(abs(km_train-trainkm)))
  b = max(max(abs(km_test-testkm)))
  if(a+b<1e-7)
    y = 0;
  else
    y = 1;
  end
