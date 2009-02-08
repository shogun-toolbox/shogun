function y = check_accuracy(accuracy, data)
	ending=sprintf('<--- accuracy: %e\n', accuracy);
	if strcmp(data{1}, 'kernel')==1 || strcmp(data{1}, 'distance')==1
		fprintf('train: %e, test: %e %s', data{2}, data{3}, ending);
	elseif strcmp(data{1}, 'classifier')==1
		fprintf('alphas: %e, bias: %e, sv: %e, classified: %e %s', ...
			data{2}, data{3}, data{4}, data{5}, ending);
	elseif strcmp(data{1}, 'kmeans')==1
		fprintf('centers: %e, radi: %e %s', data{2}, data{3}, ending);
	elseif strcmp(data{1}, 'hierarchical')==1
		fprintf('merge_distances: %e, pairs: %e %s', ...
			data{2}, data{3}, ending);
	elseif strcmp(data{1}, 'distribution')==1
		fprintf('likelihood: %e, derivatives: %e %s', ...
			data{2}, data{3}, ending);
	elseif strcmp(data{1}, 'custom')==1
		fprintf('triangletriangle: %e, fulltriangle: %e, fullfull: %e %s', ...
			data{2}, data{3}, data{4}, ending);
	end

	y=true;
	for i = 2:length(data)
		if data{i}>accuracy
			y=false;
			return;
		end
	end
