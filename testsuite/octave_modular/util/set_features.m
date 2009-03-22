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

	name=eval(sprintf([prefix, 'name']));
	feature_type=eval(sprintf([prefix, 'feature_type']));
	feature_class=eval(sprintf([prefix, 'feature_class']));
	alphabet=eval(sprintf([prefix, 'alphabet']));
	data_train=eval(sprintf([prefix, 'data_train']));
	data_test=eval(sprintf([prefix, 'data_test']));
	y=false;
	size_cache=10;

	if strcmp(name, 'Combined')==1
		% this will break when subkernels in data file are changed
		% it blindly assumes that all features are StringChar
		global kernel_subkernel0_data_train;
		global kernel_subkernel0_data_test;
		global kernel_subkernel1_data_train;
		global kernel_subkernel1_data_test;
		global kernel_subkernel2_data_train;
		global kernel_subkernel2_data_test;
		global CombinedFeatures;
		global StringCharFeatures;
		global DNA;

		feats_train=CombinedFeatures();
		feats_test=CombinedFeatures();

		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel0_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel0_data_test);
		feats_test.append_feature_obj(feat);

		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel1_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel1_data_test);
		feats_test.append_feature_obj(feat);

		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel2_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_features(kernel_subkernel2_data_test);
		feats_test.append_feature_obj(feat);

	elseif strcmp(name, 'Custom')==1
		global kernel_data;
		global RealFeatures;
		feats_train=RealFeatures(kernel_data);
		feats_test=RealFeatures(kernel_data);

	elseif strcmp(topfk_name, 'FK')==1
		global pos_train;
		global pos_test;
		global neg_train;
		global neg_test;
		global FKFeatures;

		if !set_pos_and_neg('topfk_')
			return;
		end

		feats_train=FKFeatures(size_cache, pos_train, neg_train);
		feats_train.set_opt_a(-1); %estimate prior
		feats_test=FKFeatures(size_cache, pos_test, neg_test);
		feats_test.set_a(feats_train.get_a()); %use prior from training data

	elseif strcmp(topfk_name, 'TOP')==1
		global pos_train;
		global pos_test;
		global neg_train;
		global neg_test;
		global TOPFeatures;

		if !set_pos_and_neg('topfk_')
			return;
		end

		feats_train=TOPFeatures(size_cache, pos_train, neg_train, false, false);
		feats_test=TOPFeatures(size_cache, pos_test, neg_test, false, false);

	else
		global classifier_type;
		if findstr(name, 'Sparse')
			is_sparse=true;
		elseif (!isempty(classifier_type) &&
			strcmp(classifier_type, 'linear')==1)
			is_sparse=true;
		else
			is_sparse=false;
		end

		if strcmp(feature_class, 'simple')==1
			if strcmp(feature_type, 'Real')==1
				global RealFeatures;
				feats_train=RealFeatures(double(data_train));
				feats_test=RealFeatures(double(data_test));

				if is_sparse
					global SparseRealFeatures;
					sparse=SparseRealFeatures();
					sparse.obtain_from_simple(feats_train);
					feats_train=sparse;
					sparse=SparseRealFeatures();
					sparse.obtain_from_simple(feats_test);
					feats_test=sparse;
				end

			elseif strcmp(feature_type, 'Word')==1
				global WordFeatures;
				feats_train=WordFeatures(uint16(data_train));
				feats_test=WordFeatures(uint16(data_test));

			elseif strcmp(feature_type, 'Byte')==1
				global RAWBYTE;
				global ByteFeatures;
				feats_train=ByteFeatures(RAWBYTE);
				feats_train.copy_feature_matrix(uint8(data_train));
				feats_test=ByteFeatures(RAWBYTE);
				feats_test.copy_feature_matrix(uint8(data_test));

			else
				fprintf('Simple feature type %s not supported yet!\n', feature_type);
				return;
			end
		elseif strcmp(feature_class, 'string')==1 || strcmp(feature_class, 'string_complex')==1
			global DNA;
			global CUBE;

			if strcmp(alphabet, 'DNA')==1
				alphabet=DNA;
			elseif strcmp(alphabet, 'CUBE')==1
				alphabet=CUBE;
			elseif strcmp(alphabet, 'RAWBYTE')==1
				disp('Alphabet RAWBYTE not supported yet.');
				return;
			else
				error('Alphabet %s not supported yet!', alphabet);
			end

			global StringCharFeatures;
			feats_train=StringCharFeatures(alphabet);
			feats_train.set_features(data_train);
			feats_test=StringCharFeatures(alphabet);
			feats_test.set_features(data_test);

			if strcmp(feature_class, 'string_complex')==1
				convert_features_and_add_preproc(prefix);
			end

		elseif strcmp(feature_class, 'wd')==1
			global DNA;
			global RAWDNA;
			global StringCharFeatures;
			global StringByteFeatures;
			global WDFeatures;
			eval(sprintf(['global ', prefix, 'order']));
			order=eval(sprintf([prefix, 'order']));

			charfeat=StringCharFeatures(DNA);
			charfeat.set_features(data_train);
			bytefeat=StringByteFeatures(RAWDNA);
			bytefeat.obtain_from_char(charfeat, 0, 1, 0, false);
			feats_train=WDFeatures(bytefeat, order, order);

			charfeat=StringCharFeatures(DNA);
			charfeat.set_features(data_test);
			bytefeat=StringByteFeatures(RAWDNA);
			bytefeat.obtain_from_char(charfeat, 0, 1, 0, false);
			feats_test=WDFeatures(bytefeat, order, order);

		else
			error('Unknown feature class %s', feature_class);
		end
	end

	y=true;
