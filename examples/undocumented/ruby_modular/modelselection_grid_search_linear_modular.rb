# this was trancekoded by the awesome trancekoder
# ...and fixifikated by the awesum fixifikator
require 'modshogun'
require 'pp'
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2011 Heiko Strathmann
# Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
#


# generate some overlapping training vectors
num_vectors=100
vec_distance=1
traindat=concatenate((randn(2,num_vectors)-vec_distance,
	randn(2,num_vectors)+vec_distance), axis=1)
label_traindat=concatenate((-ones(num_vectors), ones(num_vectors)));

parameter_list = [[traindat,label_traindat]]

def modelselection_grid_search_simple(traindat=traindat, label_traindat=label_traindat)

	# build parameter tree to select C1 and C2 
# *** 	param_tree_root=ModelSelectionParameters()
	param_tree_root=Modshogun::ModelSelectionParameters.new
	param_tree_root.set_features()
# *** 	c1=ModelSelectionParameters("C1");
	c1=Modshogun::ModelSelectionParameters.new
	c1.set_features("C1");
	param_tree_root.append_child(c1)
	c1.build_values(-2.0, 2.0, R_EXP);

# *** 	c2=ModelSelectionParameters("C2");
	c2=Modshogun::ModelSelectionParameters.new
	c2.set_features("C2");
	param_tree_root.append_child(c2);
	c2.build_values(-2.0, 2.0, R_EXP);

	# training data
# *** 	features=RealFeatures(traindat)
	features=Modshogun::RealFeatures.new
	features.set_features(traindat)
# *** 	labels=Labels(label_traindat)
	labels=Modshogun::Labels.new
	labels.set_features(label_traindat)

	# classifier
# *** 	classifier=LibLinear(L2R_L2LOSS_SVC)
	classifier=Modshogun::LibLinear.new
	classifier.set_features(L2R_L2LOSS_SVC)

	# splitting strategy for cross-validation
# *** 	splitting_strategy=StratifiedCrossValidationSplitting(labels, 10)
	splitting_strategy=Modshogun::StratifiedCrossValidationSplitting.new
	splitting_strategy.set_features(labels, 10)

	# evaluation method
# *** 	evaluation_criterium=ContingencyTableEvaluation(ACCURACY)
	evaluation_criterium=Modshogun::ContingencyTableEvaluation.new
	evaluation_criterium.set_features(ACCURACY)

	# cross-validation instance
# *** 	cross_validation=CrossValidation(classifier, features, labels,
	cross_validation=Modshogun::CrossValidation.new
	cross_validation.set_features(classifier, features, labels,
		splitting_strategy, evaluation_criterium)

	# model selection instance
# *** 	model_selection=GridSearchModelSelection(param_tree_root,
	model_selection=Modshogun::GridSearchModelSelection.new
	model_selection.set_features(param_tree_root,
		cross_validation)

	# perform model selection with selected methods
	puts "performing model selection of"
	param_tree_root.print_tree()
	best_parameters=model_selection.select_model()

	#	puts best parameters
	puts "best parameters:"
	best_parameters.print_tree()

	# apply them and	puts result
	best_parameters.apply_to_machine(classifier)
	result=cross_validation.evaluate()
	result.print_result()


end
if __FILE__ == $0
	puts 'GridSearchSimple'
	modelselection_grid_search_simple(*parameter_list[0])

end
