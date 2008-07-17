function y = check_accuracy_hierarchical(accuracy, merge_distances, pairs)
	printf("merge_distances: %e, pairs: %e, <--- accuracy: %e\n", merge_distances, pairs, accuracy);

	if merge_distances>accuracy
		y=false;
	elseif pairs>accuracy
		y=false;
	else
		y=true;
	end
