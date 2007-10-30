function y = test_poly_kernel(filename)

  eval(filename);

  sg('set_features', 'TRAIN', traindat);

  if(strcmp(inhom, 'True'))
      inhom='1';
  else 
      inhom='0';
  end
  
  if(strcmp(use_norm, 'True'))
     use_norm='1';
  else
     use_norm='0';
  end

  kname = ['set_kernel ', 'POLY REAL ', num2str(size_),' ', num2str(degree),' ',inhom,' ', use_norm];
%  kname = ['set_kernel ', 'POLY REAL ', num2str(degree),' ',inhom,' ', use_norm, ' ', num2str(size_),' '];

  sg('send_command',kname);

  sg('send_command', 'init_kernel TRAIN');
  trainkm = sg('get_kernel_matrix');

  sg('set_features', 'TEST', testdat);
  sg('send_command', 'init_kernel TEST');
  testkm = sg('get_kernel_matrix');

  a = max(max(abs(km_test-testkm)))
  b = max(max(abs(km_train-trainkm)))

  if a<1e-6 || b<1e-6
    y = 0;
  else
    y = 1;
  end
