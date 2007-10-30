function y = test_cws_kernel(filename)

  eval(filename);

% need to be reshaped because of suboptimal definition of data in testsuite
	seqlen = 60
	traindat = reshape(traindat, seqlen, length(traindat)/seqlen);
	testdat = reshape(testdat, seqlen, length(testdat)/seqlen);

  kname = ['set_kernel ', sprintf('COMMSTRING WORD')];

  sg('set_features', 'TRAIN', traindat, alphabet);
  sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %i %i %i %i', order, order-1, gap, reverse));
  sg('send_command', 'add_preproc SORTWORDSTRING') ;
  sg('send_command', 'attach_preproc TRAIN') ;
  sg('send_command', kname);
  sg('send_command', 'init_kernel TRAIN');
  trainkm = sg('get_kernel_matrix');


  sg('set_features', 'TEST', testdat, alphabet);
  sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %i %i %i %i', order, order-1, gap, reverse));
%  sg('send_command', 'add_preproc SORTWORDSTRING') ;
  sg('send_command', 'attach_preproc TEST') ;
  sg('send_command', kname);
  sg('send_command', 'init_kernel TEST');
  testkm = sg('get_kernel_matrix');

  a = max(max(abs(km_train-trainkm)))
  b = max(max(abs(km_test-testkm)))
  if(a+b<1e-7)
    y = 0;
  else
    y = 1;
  end
