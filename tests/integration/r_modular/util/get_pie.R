get_pie <- function(feats) {
	pie <- PluginEstimate()
	lab <- as.double(classifier_labels)
	labels <- Labels(lab)
	pie$set_labels(pie, labels)
	pie$set_features(pie, feats)
	pie$train()
	return(pie)
}
