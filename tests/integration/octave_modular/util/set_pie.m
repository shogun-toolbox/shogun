function y = set_pie()
	global classifier_labels;
	global pie;
	global PluginEstimate;
	global BinaryLabels;
	global feats_train;
	y=true;

	pie=PluginEstimate();
	labels=BinaryLabels(classifier_labels);
	pie.set_labels(labels);
	pie.set_features(feats_train);
	pie.train();
