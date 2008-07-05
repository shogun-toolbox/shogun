function y = kernel(filename)
	addpath('../../testsuite/data/kernel');

	eval(filename);
	set_features();
	%y=feval(name,filename)
	y=1;
