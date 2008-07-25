function y = set_features()
	global name;
	global name_features;
	global feature_type;
	global feature_class;
	global alphabet;
	global data_train;
	global data_test;
	global feats_train;
	global feats_test;
	y=false;
	size_cache=10;

	if strcmp(name, 'Combined')==1
		% this will break when subkernels in data file are changed
		% it blindly assumes that all features are StringChar
		global subkernel0_data_train;
		global subkernel0_data_test;
		global subkernel1_data_train;
		global subkernel1_data_test;
		global subkernel2_data_train;
		global subkernel2_data_test;
		global CombinedFeatures;
		global StringCharFeatures;
		global DNA;

		feats_train=CombinedFeatures();
		feats_test=CombinedFeatures();

		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel0_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel0_data_test);
		feats_test.append_feature_obj(feat);

		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel1_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel1_data_test);
		feats_test.append_feature_obj(feat);

		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel2_data_train);
		feats_train.append_feature_obj(feat);
		feat=StringCharFeatures(DNA);
		feat.set_string_features(subkernel2_data_test);
		feats_test.append_feature_obj(feat);

	elseif strcmp(name, 'Custom')==1
		global data;
		global RealFeatures;
		feats_train=RealFeatures(data);
		feats_test=RealFeatures(data);

	elseif strcmp(name_features, 'FK')==1
		global pos;
		global pos_clone;
		global neg;
		global neg_clone;

		if !set_pos_and_neg()
			return;
		end

		global FKFeatures;
		feats_train=FKFeatures(size_cache, pos, neg);
		feats_train.set_opt_a(-1); %estimate prior
		feats_test=FKFeatures(size_cache, pos_clone, neg_clone);
		feats_test.set_a(feats_train.get_a()); %use prior from training data

	elseif strcmp(name_features, 'TOP')==1
		global pos;
		global pos_clone;
		global neg;
		global neg_clone;

		if !set_pos_and_neg()
			return;
		end

		global TOPFeatures;
		feats_train=TOPFeatures(size_cache, pos, neg, false, false);
		feats_test=TOPFeatures(size_cache, pos_clone, neg_clone, false, false);

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
			feats_train.set_string_features(data_train);
			feats_test=StringCharFeatures(alphabet);
			feats_test.set_string_features(data_test);

			if strcmp(feature_class, 'string_complex')==1
				convert_features_and_add_preproc();
			end

		else
			error('Unknown feature class %s', feature_class);
		end
	end

	y=true;
