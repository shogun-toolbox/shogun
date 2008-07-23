function y = check_accuracy(accuracy, train, test)
	fprintf('train: %e, test: %e <--- accuracy: %e\n', train, test, accuracy);

	if train>accuracy
		y=false;
	elseif test>accuracy
		y=false;
	else
		y=true;
	end
