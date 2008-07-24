function y = check_accuracy_distribution(accuracy, likelihood, derivatives)
	fprintf('likelihood: %e, derivatives %e <--- accuracy: %e\n',
		likelihood, derivatives, accuracy);

	if likelihood>accuracy
		y=false;
	elseif derivatives>accuracy
		y=false;
	else
		y=true;
	end
