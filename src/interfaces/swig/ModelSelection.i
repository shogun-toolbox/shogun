/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Sergey Lisitsyn
 */

/* These functions return new Objects */
%newobject *::select_model();
%newobject CParameterCombination::copy_tree();
%newobject CParameterCombination::leaf_sets_multiplication();
%newobject CModelSelectionParameters::get_combinations();
%newobject CModelSelectionParameters::get_single_combination();

/* what about parameter_set_multiplication returns new DynArray<Parameter*>? */

/* Remove C Prefix */
%rename(GridSearchModelSelection) CGridSearchModelSelection;
%rename(RandomSearchModelSelection) CRandomSearchModelSelection;
#ifdef USE_GPL_SHOGUN
%rename(GradientModelSelection) CGradientModelSelection;
#endif //USE_GPL_SHOGUN
%rename(ModelSelectionBase) CModelSelection;
%rename(ModelSelectionParameters) CModelSelectionParameters;
%rename(ParameterCombination) CParameterCombination;

%include <shogun/modelselection/ModelSelection.h>
%include <shogun/modelselection/GridSearchModelSelection.h>
%include <shogun/modelselection/RandomSearchModelSelection.h>
%include <shogun/modelselection/ParameterCombination.h>
%include <shogun/modelselection/ModelSelectionParameters.h>
#ifdef USE_GPL_SHOGUN
%include <shogun/modelselection/GradientModelSelection.h>
#endif //USE_GPL_SHOGUN
