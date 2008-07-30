function y = classifier(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/classifier');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if ~set_features()
		return;
	end

	if strcmp(classifier_type, 'kernel')==1
		if ~set_kernel()
			return;
		end
	elseif strcmp(classifier_type, 'knn')==1
		if ~set_distance()
			return;
		end
	end

	if ~isempty(classifier_labels)
		lab=Labels(classifier_labels);
	end

	if strcmp(name, 'GMNPSVM')==1
		classifier=GMNPSVM(classifier_C, kernel, lab);

	elseif strcmp(name, 'GPBTSVM')==1
		classifier=GPBTSVM(classifier_C, kernel, lab);

	elseif strcmp(name, 'KNN')==1
		classifier=KNN(classifier_k, distance, lab);

	elseif strcmp(name, 'LDA')==1
		classifier=LDA(classifier_gamma, feats_train, lab);

	elseif strcmp(name, 'LibLinear')==1
		classifier=LibLinear(classifier_C, feats_train, lab);

	elseif strcmp(name, 'LibSVMMultiClass')==1
		classifier=LibSVMMultiClass(classifier_C, kernel, lab);

	elseif strcmp(name, 'LibSVMOneClass')==1
		classifier=LibSVMOneClass(classifier_C, kernel);

	elseif strcmp(name, 'LibSVM')==1
		classifier=LibSVM(classifier_C, kernel, lab);

	elseif strcmp(name, 'MPDSVM')==1
		classifier=MPDSVM(classifier_C, kernel, lab);

	elseif strcmp(name, 'Perceptron')==1
		classifier=Perceptron(feats_train, lab);
		classifier.set_learn_rate(classifier_learn_rate);
		classifier.set_max_iter(classifier_max_iter);

	elseif strcmp(name, 'SVMLight')==1
		try
			classifier=SVMLight(classifier_C, kernel, lab);
		catch
			disp('No support for SVMLight available.');
			return;
		end

	elseif strcmp(name, 'SVMLin')==1
		classifier=SVMLin(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SVMOcas')==1
		classifier=SVMOcas(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SVMSGD')==1
		classifier=SVMSGD(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SubGradientSVM')==1
		classifier=SubGradientSVM(classifier_C, feats_train, lab);

	else
		error('Unsupported classifier %s', name);
	end

	classifier.parallel.set_num_threads(classifier_num_threads);
	if strcmp(classifier_type, 'linear')==1 && ~isempty(classifier_bias)
		classifier.set_bias_enabled(true);
	end
	if ~isempty(classifier_epsilon) && strcmp(name, 'SVMSGD')!=1
		classifier.set_epsilon(classifier_epsilon);
	end
	if ~isempty(classifier_tube_epsilon)
		classifier.set_tube_epsilon(classifier_tube_epsilon);
	end
	if ~isempty(classifier_max_train_time)
		classifier.set_max_train_time(classifier_max_train_time);
	end
	if ~isempty(classifier_linadd_enabled)
		classifier.set_linadd_enabled(tobool(classifier_linadd_enabled));
	end
	if ~isempty(classifier_batch_enabled)
		classifier.set_batch_computation_enabled(
			tobool(classifier_batch_enabled));
	end

	classifier.train();

	alphas=0;
	bias=0;
	sv=0;
	if ~isempty(classifier_bias)
		bias=classifier.get_bias();
		bias=abs(bias-classifier_bias);
	end
	if ~isempty(classifier_alphas)
		alphas=classifier.get_alphas();
		alphas=max(abs(alphas-classifier_alphas));
	end
	if ~isempty(classifier_support_vectors)
		sv=classifier.get_support_vectors();
		sv=max(abs(sv-classifier_support_vectors));
	end

	if strcmp(classifier_type, 'knn')==1
		distance.init(feats_train, feats_test);
	elseif strcmp(classifier_type, 'kernel')==1
		kernel.init(feats_train, feats_test);
	elseif (strcmp(classifier_type, 'lda')==1 ||
		strcmp(classifier_type, 'linear')==1 ||
		strcmp(classifier_type, 'perceptron')==1)
		classifier.set_features(feats_test);
	end

	classified=max(abs(
		classifier.classify().get_labels()-classifier_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(classifier_accuracy, data);
