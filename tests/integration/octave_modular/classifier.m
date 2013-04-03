function y = classifier(filename)
	init_shogun;
	y=true;
	addpath('util');
	addpath('../data/classifier');

	eval('globals'); % ugly hack to have vars from filename as globals
	eval(filename);

	if strcmp(classifier_type, 'kernel')==1
		if ~set_features('kernel_')
			return;
		end
		if ~set_kernel()
			return;
		end
	elseif strcmp(classifier_type, 'knn')==1
		if ~set_features('distance_')
			return;
		end
		if ~set_distance()
			return;
		end
	else
		if ~set_features('classifier_')
			return;
		end
	end

	if ~isempty(classifier_labels)
		lab=BinaryLabels(classifier_labels);
	end

	if strcmp(classifier_name, 'GMNPSVM')==1
		classifier=GMNPSVM(classifier_C, kernel, lab);

	elseif strcmp(classifier_name, 'GPBTSVM')==1
		classifier=GPBTSVM(classifier_C, kernel, lab);

	elseif strcmp(classifier_name, 'KNN')==1
		classifier=KNN(classifier_k, distance, lab);

	elseif strcmp(classifier_name, 'LDA')==1
		classifier=LDA(classifier_gamma, feats_train, lab);

	elseif strcmp(classifier_name, 'LibLinear')==1
		classifier=LibLinear(classifier_C, feats_train, lab);

	elseif strcmp(classifier_name, 'LibSVMMultiClass')==1
		classifier=LibSVMMultiClass(classifier_C, kernel, lab);
		classifier.set_solver_type(L2R_LR);

	elseif strcmp(classifier_name, 'LibSVMOneClass')==1
		classifier=LibSVMOneClass(classifier_C, kernel);

	elseif strcmp(classifier_name, 'LibSVM')==1
		classifier=LibSVM(classifier_C, kernel, lab);

	elseif strcmp(classifier_name, 'MPDSVM')==1
		classifier=MPDSVM(classifier_C, kernel, lab);

	elseif strcmp(classifier_name, 'Perceptron')==1
		classifier=Perceptron(feats_train, lab);
		classifier.set_learn_rate(classifier_learn_rate);
		classifier.set_max_iter(classifier_max_iter);

	elseif strcmp(classifier_name, 'SVMLight')==1
		try
			classifier=SVMLight(classifier_C, kernel, lab);
		catch
			disp('No support for SVMLight available.');
			return;
		end

	elseif strcmp(classifier_name, 'SVMLin')==1
		classifier=SVMLin(classifier_C, feats_train, lab);

	elseif strcmp(classifier_name, 'SVMOcas')==1
		classifier=SVMOcas(classifier_C, feats_train, lab);
		classifier.set_bias_enabled(false);

	elseif strcmp(classifier_name, 'SVMSGD')==1
		classifier=SVMSGD(classifier_C, feats_train, lab);

	elseif strcmp(classifier_name, 'SubGradientSVM')==1
		classifier=SubGradientSVM(classifier_C, feats_train, lab);

	else
		error('Unsupported classifier %s', classifier_name);
	end

	classifier.parallel.set_num_threads(classifier_num_threads);
	if strcmp(classifier_type, 'linear')==1 && ~isempty(classifier_bias)
		classifier.set_bias_enabled(true);
	end
	if ~isempty(classifier_epsilon) && strcmp(classifier_name, 'SVMSGD')!=1
		classifier.set_epsilon(classifier_epsilon);
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

	bias=0;
	if ~isempty(classifier_bias)
		bias=classifier.get_bias();
		bias=abs(bias-classifier_bias);
	end

	alphas=0;
	sv=0;
	if ~isempty(classifier_alpha_sum) && ~isempty(classifier_sv_sum)
		if strcmp(classifier_label_type, 'series')==1
			for i = 0:classifier.get_num_svms()-1
				subsvm=classifier.get_svm(i);
				tmp=subsvm.get_alphas();
				for j = 1:length(tmp)
					alphas=alphas+tmp(j:j);
				end
				tmp=subsvm.get_support_vectors();
				for j = 1:length(tmp)
					sv=sv+tmp(j:j);
				end
			end
			alphas=abs(alphas-classifier_alpha_sum);
			sv=abs(sv-classifier_sv_sum);
		else
			tmp=classifier.get_alphas();
			for i = 1:length(tmp)
				alphas=alphas+tmp(i:i);
			end
			alphas=abs(alphas-classifier_alpha_sum);
			tmp=classifier.get_support_vectors();
			for i = 1:length(tmp)
				sv=sv+tmp(i:i);
			end
			sv=abs(sv-classifier_sv_sum);
		end
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
		classifier.apply().get_values()-classifier_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(classifier_accuracy, data);
