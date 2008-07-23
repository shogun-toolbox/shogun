function y = set_pie()
	global classifier_labels;
	global pie;

	pie=PluginEstimate();
	labels=Labels(classifier_labels);
	pie.set_labels(labels);
	pie.set_features(feats_train);
	pie.train();

	y=true;
