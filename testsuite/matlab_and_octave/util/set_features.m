function y = set_features(prefix)
	eval(sprintf(['global ', prefix, 'name']));
	eval(sprintf(['global ', prefix, 'feature_type']));
	eval(sprintf(['global ', prefix, 'feature_class']));
	eval(sprintf(['global ', prefix, 'alphabet']));
	eval(sprintf(['global ', prefix, 'data_train']));
	eval(sprintf(['global ', prefix, 'data_test']));
	global feats_train;
	global feats_test;
	global topfk_name;
	global kernel_data;
	global classifier_type;

	name=eval(sprintf([prefix, 'name']));
	feature_type=eval(sprintf([prefix, 'feature_type']));
	feature_class=eval(sprintf([prefix, 'feature_class']));
	alphabet=eval(sprintf([prefix, 'alphabet']));
	data_train=eval(sprintf([prefix, 'data_train']));
	data_test=eval(sprintf([prefix, 'data_test']));
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

	if ~isempty(topfk_name)
		fprintf('Features %s not yet supported!\n', topfk_name);
		return;
	end

	if strcmp(name, 'Combined')==1
		global kernel_subkernel0_alphabet;
		global kernel_subkernel0_data_train;
		global kernel_subkernel0_data_test;
		global kernel_subkernel1_alphabet;
		global kernel_subkernel1_data_train;
		global kernel_subkernel1_data_test;
		global kernel_subkernel2_alphabet;
		global kernel_subkernel2_data_train;
		global kernel_subkernel2_data_test;

		if isempty(kernel_subkernel0_alphabet)
			sg('add_features', 'TRAIN', kernel_subkernel0_data_train);
			sg('add_features', 'TEST', kernel_subkernel0_data_test);
		else
			sg('add_features', 'TRAIN', ...
				kernel_subkernel0_data_train, kernel_subkernel0_alphabet);
			sg('add_features', 'TEST', ...
				kernel_subkernel0_data_test, kernel_subkernel0_alphabet);
		end

		if isempty(kernel_subkernel1_alphabet)
			sg('add_features', 'TRAIN', kernel_subkernel1_data_train);
			sg('add_features', 'TEST', kernel_subkernel1_data_test);
		else
			sg('add_features', 'TRAIN', ...
				kernel_subkernel1_data_train, kernel_subkernel1_alphabet);
			sg('add_features', 'TEST', ...
				kernel_subkernel1_data_test, kernel_subkernel1_alphabet);
		end

		if isempty(kernel_subkernel2_alphabet)
			sg('add_features', 'TRAIN', kernel_subkernel2_data_train);
			sg('add_features', 'TEST', kernel_subkernel2_data_test);
		else
			sg('add_features', 'TRAIN', ...
				kernel_subkernel2_data_train, kernel_subkernel2_alphabet);
			sg('add_features', 'TEST', ...
				kernel_subkernel2_data_test, kernel_subkernel2_alphabet);
		end

	elseif ~isempty(alphabet)
		if strcmp(alphabet, 'RAWDNA')==1
			sg('set_features', 'TRAIN', data_train, 'DNA');
			sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'BYTE');
			sg('set_features', 'TEST', data_test, 'DNA');
			sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'BYTE');
		else
			sg('set_features', 'TRAIN', data_train, alphabet);
			sg('set_features', 'TEST', data_test, alphabet);
		end

	elseif ~isempty(kernel_data)
		sg('set_features', 'TRAIN', kernel_data);
		sg('set_features', 'TEST', kernel_data);

	else
		fname='double';
		if strcmp(feature_type, 'Word')==1
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

	convert_features_and_add_preproc(prefix);
	y=true;
