function y = test_octave(filename)
	addpath('util');

	slashes=findstr('/', filename);
	pos_filename=slashes(end);
	pos_modulename=slashes(end-1);
	modulename=filename(pos_modulename+1:pos_filename-1);
	filename=filename(pos_filename+1:end-2);
	testcase=strcat(modulename, '("', filename, '")');
	res=eval(testcase);

	if (res==0)
		y=0;
	else
		y=1;
	end
