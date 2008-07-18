function y = check_accuracy(accuracy, likelihood)
	printf("likelihood: %e <--- accuracy: %e\n", likelihood, accuracy);

	if likelihood>accuracy
		y=false;
	else
		y=true;
	end
