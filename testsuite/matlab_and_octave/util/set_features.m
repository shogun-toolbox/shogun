function y = set_features()
	global name;
	global classifier_type;
	global alphabet;
	global data_train;
	global data_test;
	global data_type;
	global data;

	if findstr('Sparse', name)
		printf("Sparse features not supported yet\n");
		y=1;
		return
	end

	if findstr('linear', classifier_type)
		printf("Linear classifiers with sparse features not supported yet.\n");
		y=1;
		return
	end

	if strcmp(alphabet, 'RAWBYTE')==1
		fprintf(1, "Alphabet RAWBYTE not supported yet.\n");
		y=1;
		return
	end

	if strcmp(name, 'Combined')==1
		y=0;
		return
	end

	if !isempty(alphabet)
		sg('set_features', 'TRAIN', data_train, alphabet);
		sg('set_features', 'TEST', data_test, alphabet);
	elseif !isempty(data)
		sg('set_features', 'TRAIN', data);
		sg('set_features', 'TEST', data);
	else
		fname='double';
		if strcmp(data_type, 'ushort')==1
			fname='uint16';
		end

		if iscell(data_train)
			data_train=cellfun(@str2num, data_train);
			data_test=cellfun(@str2num, data_test);
		end

		sg('set_features', 'TRAIN', feval(fname, data_train));
		sg('set_features', 'TEST', feval(fname, data_test));
	end

	convert_features_and_add_preproc();
	y=0;
