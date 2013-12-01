/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
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
%rename(GradientModelSelection) CGradientModelSelection;
%rename(ModelSelectionBase) CModelSelection;
%rename(ModelSelectionParameters) CModelSelectionParameters;
%rename(ParameterCombination) CParameterCombination;

%include <shogun/modelselection/ModelSelection.h>
%include <shogun/modelselection/GridSearchModelSelection.h>
%include <shogun/modelselection/RandomSearchModelSelection.h>
%include <shogun/modelselection/ParameterCombination.h>
%include <shogun/modelselection/ModelSelectionParameters.h>
%include <shogun/modelselection/GradientModelSelection.h>
