function y = classifier(filename)
	addpath('util');
	addpath('../data/classifier');
	y=true;

	eval('globals'); % ugly hack to have vars from filename as globals
	%
	%system(sprintf('ln -sf ../data/classifier/%s.m testscript.m', filename));
	%testscript;
	%system('rm -f testscript.m'); %avoid ultra long filenames (>63 chars)
	eval(filename);

	% b0rked, skip these
	if strcmp(classifier_name, 'Perceptron')==1 || strcmp(classifier_name, 'SubGradientSVM')==1
		fprintf('%s currently does not have nice data.\n', classifier_name);
		return;
	end

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
		sg('set_labels', 'TRAIN', classifier_labels);
	end

	if strcmp(classifier_name, 'SVMOcas')==1
		sg('svm_use_bias', false);
	end

	cname=fix_classifier_name_inconsistency(classifier_name);
	try
		sg('new_classifier', cname);
	catch
		fprintf('Cannot set classifier %s!\n', cname);
		return;
	end

	if ~isempty(classifier_bias_enabled)
		sg('svm_use_bias', tobool(classifier_bias_enabled));
	end
	if ~isempty(classifier_epsilon)
		sg('svm_epsilon', classifier_epsilon);
	end
	if ~isempty(classifier_max_train_time)
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
			sg('c', classifier_C);
		end
		sg('train_classifier');
	end

	alphas=0;
	bias=0;
	sv=0;

	if strcmp(classifier_type, 'lda')==1
		0; % nop
	else
		if ~isempty(classifier_bias) && strcmp(classifier_label_type, 'series')~=1
			[bias, weights]=sg('get_svm');
			bias=abs(bias-classifier_bias);
		end

		if ~isempty(classifier_alpha_sum) && ~isempty(classifier_sv_sum)
			if strcmp(classifier_label_type, 'series')==1
				for i = 0:sg('get_num_svms')-1
					[dump, weights]=sg('get_svm', i);
					weights=weights';
					for j = 1:length(weights(1:1, :))
						alphas=alphas+weights(1:1, j:j);
					end
					for j = 1:length(weights(2:2, :))
						sv=sv+weights(2:2, j:j);
					end
				end
				alphas=abs(alphas-classifier_alpha_sum);
				sv=abs(sv-classifier_sv_sum);
			else
				[dump, weights]=sg('get_svm');
				weights=weights';
				for i = 1:length(weights(1:1, :))
					alphas=alphas+weights(1:1, i:i);
				end
				alphas=abs(alphas-classifier_alpha_sum);
				for i = 1:length(weights(2:2, :))
					sv=sv+weights(2:2, i:i);
				end
				sv=abs(sv-classifier_sv_sum);
			end
		end
	end

	if strcmp(classifier_name, 'WDSVMOcas')==1
		converted=sg('get_features', 'TRAIN');
		for i = 1:length(classifier_data_train)
			[classifier_data_train(i), converted(i)]
		end
		classified=sg('classify')
		classifier_classified
	end

	classified=max(abs(sg('classify')-classifier_classified));

	data={'classifier', alphas, bias, sv, classified};
	y=check_accuracy(classifier_accuracy, data);
