function y = set_features()
	global name;
	global classifier_type;
	global alphabet;
	global data_train;
	global data_test;
	global data;

	if findstr('Sparse', name)
		fprintf(1, 'Sparse features not supported yet');
		y=1;
		return
	end

	if findstr('linear', classifier_type)
		fprintf(1, 'Linear classifiers with sparse features not supported yet');
		y=1;
		return
	end

	if findstr('RAWBYTE', alphabet)
		fprintf(1, 'Alphabet RAWBYTE not supported yed.');
		y=1;
		return
	end

	if !findstr('Combined', name)
		if !isempty(alphabet)
			sg('set_features', 'TRAIN', data_train, alphabet);
			sg('set_features', 'TEST', data_test, alphabet);
		elseif !isempty(data)
			sg('set_features', 'TRAIN', data);
			sg('set_features', 'TEST', data);
		else
			sg('set_features', 'TRAIN', data_train);
			sg('set_features', 'TEST', data_test);
		end
	end

	convert_features_and_add_preproc();
	y=0;
