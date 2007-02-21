function y = test_poly_kernel(filename)

  eval(filename);

  sg('set_features', 'TRAIN', traindat);

  if(length(inhom)==length('True'))
    if(inhom=='True')
      inhom='1';
    else 
      inhom='0';
    end
  else 
      inhom='0';
  end
  
  if(length(use_norm)==length('True'))
     if(use_norm=='True')
        use_norm='1';
     else 
       use_norm='0';
     end
  else
     use_norm='0';
  end

  kname = ['set_kernel ', 'POLY REAL ', num2str(size_),' ', num2str(degree),' ',inhom,' ', use_norm];

  sg('send_command',kname);
 

  sg('send_command', 'init_kernel TRAIN');
  trainkm = sg('get_kernel_matrix');

  sg('set_features', 'TEST', testdat);
  sg('send_command', 'init_kernel TEST');
  testkm = sg('get_kernel_matrix');
  
  

  orgkm = km_test
  newkm = testkm
  

  a = max(max(abs(km_test-testkm)))
  b = max(max(abs(km_train-trainkm)))
  if(a+b<1e-7)
    y = 0;
  else
    y = 1;
  end