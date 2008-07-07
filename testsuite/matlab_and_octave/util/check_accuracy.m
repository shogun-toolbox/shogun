function y = check_accuracy(accuracy, ktrain, ktest)
	printf("ktrain: %e, ktest: %e <--- accuracy: %e\n", ktrain, ktest, accuracy);

	if ktrain>accuracy
		y=1;
	elseif ktest>accuracy
		y=1;
	else
		y=0;
	end
