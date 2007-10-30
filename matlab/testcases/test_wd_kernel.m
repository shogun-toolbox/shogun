function y = test_wd_kernel(filename)
	eval(filename);

% need to be reshaped because of suboptimal definition of data in testsuite
	traindat = reshape(traindat, seqlen, length(traindat)/seqlen);
	testdat = reshape(testdat, seqlen, length(testdat)/seqlen);

	sg('set_features', 'TRAIN', traindat, alphabet);
	sg('send_command',sprintf('set_kernel WEIGHTEDDEGREE STRING 10 %i',degree));
	sg('send_command', 'init_kernel TRAIN');
	%set_subkernel_weights missing
	trainkm = sg('get_kernel_matrix');
	a = max(max(abs(km_train-trainkm)));

	sg('set_features', 'TEST', testdat, alphabet);
	sg('send_command', 'init_kernel TEST');
	testkm = sg('get_kernel_matrix');
	b = max(max(abs(km_test-testkm)));

	if(a+b<1e-7)
		y = 0;
	else
		y = 1;
	end
