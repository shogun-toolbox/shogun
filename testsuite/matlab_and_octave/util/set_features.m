function y = set_features()
	global name;
	global name_features;
	global classifier_type;
	global alphabet;
	global data_train;
	global data_test;
	global data_type;
	global feature_type;
	global data;
	y=false;

	if findstr('Sparse', name)
		if strcmp(feature_type, 'Real')~=1
			fprintf('Sparse features other than Real not supported yet!\n');
			return;
		end
	end

	if strcmp(alphabet, 'RAWBYTE')==1
		fprintf('Alphabet RAWBYTE not supported yet!\n');
		return;
	end

	if ~isempty(name_features)
		fprintf('Features %s not yet supported!\n', name_features);
		return;
	end

	if strcmp(name, 'Combined')==1
		global subkernel0_alphabet;
		global subkernel0_data_train;
		global subkernel0_data_test;
		global subkernel1_alphabet;
		global subkernel1_data_train;
		global subkernel1_data_test;
		global subkernel2_alphabet;
		global subkernel2_data_train;
		global subkernel2_data_test;

		if isempty(subkernel0_alphabet)
			sg('add_features', 'TRAIN', subkernel0_data_train);
			sg('add_features', 'TEST', subkernel0_data_test);
		else
			sg('add_features', 'TRAIN', ...
				subkernel0_data_train, subkernel0_alphabet);
			sg('add_features', 'TEST', ...
				subkernel0_data_test, subkernel0_alphabet);
		end

		if isempty(subkernel1_alphabet)
			sg('add_features', 'TRAIN', subkernel1_data_train);
			sg('add_features', 'TEST', subkernel1_data_test);
		else
			sg('add_features', 'TRAIN', ...
				subkernel1_data_train, subkernel1_alphabet);
			sg('add_features', 'TEST', ...
				subkernel1_data_test, subkernel1_alphabet);
		end

		if isempty(subkernel2_alphabet)
			sg('add_features', 'TRAIN', subkernel2_data_train);
			sg('add_features', 'TEST', subkernel2_data_test);
		else
			sg('add_features', 'TRAIN', ...
				subkernel2_data_train, subkernel2_alphabet);
			sg('add_features', 'TEST', ...
				subkernel2_data_test, subkernel2_alphabet);
		end

	elseif ~isempty(alphabet)
		sg('set_features', 'TRAIN', data_train, alphabet);
		sg('set_features', 'TEST', data_test, alphabet);

	elseif ~isempty(data)
		sg('set_features', 'TRAIN', data);
		sg('set_features', 'TEST', data);

	else
		fname='double';
		if strcmp(data_type, 'ushort')==1
			fname='uint16';
		elseif strcmp(classifier_type, 'linear')==1
			fname='sparse';
		elseif findstr('Sparse', name)
			fname='sparse';
		end

		if iscell(data_train)
			data_train=cellfun(@str2num, data_train);
			data_test=cellfun(@str2num, data_test);
		end

		sg('set_features', 'TRAIN', feval(fname, data_train));
		sg('set_features', 'TEST', feval(fname, data_test));
	end

	convert_features_and_add_preproc();
	y=true;
