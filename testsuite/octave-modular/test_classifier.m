function y = test_classifier(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/classifier');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	%if strcmp(name, 'Perceptron')==1 % b0rked, skip it
%		return;
%	end

	if !set_features()
		return;
	end

	if strcmp(classifier_type, 'kernel')==1
		if !set_kernel()
			return;
		end
	elseif strcmp(classifier_type, 'knn')==1
		if !set_distance()
			return;
		end
	end

	if !isempty(classifier_labels)
		lab=Labels(classifier_labels);
	end

	if strcmp(name, 'GMNPSVM')==1
		classi=GMNPSVM(classifier_C, kern, lab);

	elseif strcmp(name, 'GPBTSVM')==1
		classi=GPBTSVM(classifier_C, kern, lab);

	elseif strcmp(name, 'KNN')==1
		classi=KNN(classifier_k, dist, lab);

	elseif strcmp(name, 'LDA')==1
		classi=LDA(classifier_gamma, feats_train, lab);

	elseif strcmp(name, 'LibLinear')==1
		classi=LibLinear(classifier_C, feats_train, lab);

	elseif strcmp(name, 'LibSVMMultiClass')==1
		classi=LibSVMMultiClass(classifier_C, kern, lab);

	elseif strcmp(name, 'LibSVMOneClass')==1
		classi=LibSVMOneClass(classifier_C, kern);

	elseif strcmp(name, 'LibSVM')==1
		classi=LibSVM(classifier_C, kern, lab);

	elseif strcmp(name, 'MPDSVM')==1
		classi=MPDSVM(classifier_C, kern, lab);

	elseif strcmp(name, 'Perceptron')==1
		classi=Perceptron(feats_train, lab);
		classi.set_learn_rate(classifier_learn_rate);
		classi.set_max_iter(classifier_max_iter);

	elseif strcmp(name, 'SVMLight')==1
		classi=SVMLight(classifier_C, kern, lab);

	elseif strcmp(name, 'SVMLin')==1
		classi=SVMLin(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SVMOcas')==1
		classi=SVMOcas(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SVMSGD')==1
		classi=SVMSGD(classifier_C, feats_train, lab);

	elseif strcmp(name, 'SubGradientSVM')==1
		classi=SubGradientSVM(classifier_C, feats_train, lab);

	else
		error('Unsupported classifier %s', name);
	end

	classi.parallel.set_num_threads(classifier_num_threads);
	if strcmp(classifier_type, 'linear')==1 && !isempty(classifier_bias)
		classi.set_bias_enabled(true);
	end
	if !isempty(classifier_epsilon) && strcmp(name, 'SVMSGD')!=1
		classi.set_epsilon(classifier_epsilon);
	end
	if !isempty(classifier_tube_epsilon)
		classi.set_tube_epsilon(classifier_tube_epsilon);
	end
	if !isempty(classifier_max_train_time)
		classi.set_max_train_time(classifier_max_train_time);
	end
	if !isempty(classifier_linadd_enabled)
		classi.set_linadd_enabled(tobool(classifier_linadd_enabled));
	end
	if !isempty(classifier_batch_enabled)
		classi.set_batch_computation_enabled(
			tobool(classifier_batch_enabled));
	end

	classi.train();

	alphas=0;
	bias=0;
	sv=0;
	if !isempty(classifier_bias)
		bias=classi.get_bias();
		bias=abs(bias-classifier_bias);
	end
	if !isempty(classifier_alphas)
		alphas=classi.get_alphas();
		alphas=max(abs(alphas-classifier_alphas));
	end
	if !isempty(classifier_support_vectors)
		sv=classi.get_support_vectors();
		sv=max(abs(sv-classifier_support_vectors));
	end

	if strcmp(classifier_type, 'knn')==1
		dist.init(feats_train, feats_test);
	elseif strcmp(classifier_type, 'kernel')==1
		kern.init(feats_train, feats_test);
	elseif (strcmp(classifier_type, 'lda')==1 ||
		strcmp(classifier_type, 'linear')==1 ||
		strcmp(classifier_type, 'perceptron')==1)
		classi.set_features(feats_test);
	end

	classified=max(abs(
		classi.classify().get_labels()-classifier_classified));

	y=check_accuracy_classifier(classifier_accuracy,
		alphas, bias, sv, classified);
