function y = check_accuracy_distribution(accuracy, likelihood)
	fprintf('likelihood: %e <--- accuracy: %e\n', likelihood, accuracy);

	if likelihood>accuracy
		y=false;
	else
		y=true;
	end
