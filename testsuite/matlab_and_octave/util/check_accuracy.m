function y = check_accuracy(accuracy, ktrain, ktest)
	printf("ktrain: %e, ktest: %e <--- accuracy: %e\n", ktrain, ktest, accuracy);

	if ktrain>accuracy
		y=false;
	elseif ktest>accuracy
		y=false;
	else
		y=true;
	end
