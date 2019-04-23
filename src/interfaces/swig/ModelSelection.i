/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* These functions return new Objects */
%newobject *::select_model();
%newobject CParameterCombination::copy_tree();
%newobject CParameterCombination::leaf_sets_multiplication();
%newobject ModelSelectionParameters::get_combinations();
%newobject ModelSelectionParameters::get_single_combination();

/* what about parameter_set_multiplication returns new DynArray<Parameter*>? */

%shared_ptr(shogun::ModelSelection)
%shared_ptr(shogun::GridSearchModelSelection)
%shared_ptr(shogun::RandomSearchModelSelection)
#ifdef USE_GPL_SHOGUN
%shared_ptr(shogun::GradientModelSelection)
#endif //USE_GPL_SHOGUN
%shared_ptr(shogun::ModelSelectionBase)
%shared_ptr(shogun::ModelSelectionParameters)
%shared_ptr(shogun::ParameterCombination)

%include <shogun/modelselection/ModelSelection.h>
%include <shogun/modelselection/GridSearchModelSelection.h>
%include <shogun/modelselection/RandomSearchModelSelection.h>
%include <shogun/modelselection/ParameterCombination.h>
%include <shogun/modelselection/ModelSelectionParameters.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/modelselection/GradientModelSelection.h>
#endif //USE_GPL_SHOGUN
