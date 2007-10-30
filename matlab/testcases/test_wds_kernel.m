function y = test_wds_kernel(filename)
	eval(filename);
	max_mismatch = 0;
	shifts = sprintf('%i ', ones(seqlen, 1));

% need to be reshaped because of suboptimal definition of data in testsuite
	traindat = reshape(traindat, seqlen, length(traindat)/seqlen);
	testdat = reshape(testdat, seqlen, length(testdat)/seqlen);

	kname = sprintf('set_kernel WEIGHTEDDEGREEPOS2 STRING 10 %i %i %i %s', degree, max_mismatch, seqlen, shifts);
	sg('set_features', 'TRAIN', traindat, alphabet);
	sg('send_command', kname);
	sg('send_command', 'init_kernel TRAIN');
	%set_subkernel_weights missing
% then set weights
%trainw=rand(size(traindat)) ;
%sg('set_WD_position_weights', trainw, 'TRAIN') ;
%sg('set_WD_position_weights', trainw, 'TEST') ;
	trainkm = sg('get_kernel_matrix')
	a = max(max(abs(km_train-trainkm)));
	km_train

	sg('set_features', 'TEST', testdat, alphabet);
	sg('send_command', 'init_kernel TEST');
	testkm = sg('get_kernel_matrix');
	b = max(max(abs(km_test-testkm)));


a
b

	if(a+b<1e-7)
		y = 0;
	else
		y = 1;
	end
