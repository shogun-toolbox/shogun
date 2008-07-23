function y = check_accuracy_kmeans(accuracy, radi, centers)
	fprintf('radi: %e, centers: %e, <--- accuracy: %e\n', radi, centers, accuracy);

	if radi>accuracy
		y=false;
	elseif centers>accuracy
		y=false;
	else
		y=true;
	end
