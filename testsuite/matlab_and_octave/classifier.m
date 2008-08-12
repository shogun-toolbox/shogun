function y = classifier(filename)
	addpath('util');
	addpath('../data/classifier');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	%
	system(sprintf('ln -sf ../data/classifier/%s.m testscript.m', filename));
	testscript;
	system('rm -f testscript.m'); %avoid ultra long filenames (>63 chars)
	%eval(filename);

	if strcmp(name, 'Perceptron')==1 % b0rked, skip it
		return;
	end

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
		classifier_labels
		sg('set_labels', 'TRAIN', classifier_labels);
	end

	cname=fix_classifier_name_inconsistency(name);
	cname
	try
		sg('new_classifier', cname);
	catch
		fprintf('Cannot set classifier %s!\n', cname);
		return;
	end

	if ~isempty(classifier_bias_enabled)
		classifier_bias_enabled
		sg('svm_use_bias', tobool(classifier_bias_enabled));
	end

	if ~isempty(classifier_epsilon)
		classifier_epsilon
		sg('svm_epsilon', classifier_epsilon);
	end
	if ~isempty(classifier_tube_epsilon)
		sg('svr_tube_epsilon', classifier_tube_epsilon);
	end
	if ~isempty(classifier_max_train_time)
		classifier_max_train_time
		sg('svm_max_train_time', classifier_max_train_time);
	end
	if ~isempty(classifier_linadd_enabled)
		sg('use_linadd', true);
	end
	if ~isempty(classifier_batch_enabled)
		sg('use_batch_computation', true);
	end
	if ~isempty(classifier_num_threads)
		sg('threads', classifier_num_threads);
	end

	if strcmp(classifier_type, 'knn')==1
		sg('train_classifier', classifier_k);
	elseif strcmp(classifier_type, 'lda')==1
		sg('train_classifier', classifier_gamma);
	else
		if ~isempty(classifier_C)
			classifier_C
			sg('c', classifier_C);
		end
		sg('train_classifier');
	end

	alphas=0;
	bias=0;
	sv=0;

	if strcmp(classifier_type, 'knn')==1
		sg('init_distance', 'TEST');
	elseif strcmp(classifier_type, 'lda')==1
		0; % nop
	else
		if ~isempty(classifier_bias) && ~isempty(classifier_alphas)
			[bias, weights]=sg('get_svm');
			bias=abs(bias-classifier_bias);
			weights=weights';
			alphas=max(abs(weights(1:1,:)-classifier_alphas));
			sv=max(abs(weights(2:2,:)-classifier_support_vectors));
		end

		if strcmp(classifier_type, 'linear')~=1
			sg('init_kernel', 'TEST');
		end
	end

	%res=sg('classify')
	%classifier_classified
	%testf=sg('get_features', 'TEST')
	%trainf=sg('get_features', 'TRAIN')
	classified=max(abs(sg('classify')-classifier_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(classifier_accuracy, data);
