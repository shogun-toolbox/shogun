/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
 
%define DOCSTR
"The `Modelselection` module gathers all model selection stuff available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Modelselection
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
#%include "Modelselection_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Modelselection_includes.i"

%import "Modelselection.i"

/* These functions return new Objects */
%newobject select_model();
%newobject copy_tree();
%newobject leaf_sets_multiplication();
%newobject get_combinations();
/* what about parameter_set_multiplication returns new DynArray<Parameter*>? */
%newobject select_model();

/* Remove C Prefix */
%rename(GridSearchModelSelection) CGridSearchModelSelection;
%rename(ModelSelection) CModelSelection;
%rename(ModelSelectionParameters) CModelSelectionParameters;
%rename(ParameterCombination) CParameterCombination;

%include <shogun/modelselection/ModelSelection.h>
%include <shogun/modelselection/ModelSelectionParameters.h>
%include <shogun/modelselection/GridSearchModelSelection.h>
%include <shogun/modelselection/ParameterCombination.h>

/* Templated Class DynamicObjectArray */
%include <shogun/lib/DynamicObjectArray.h>
namespace shogun
{
    %template(DynamicParameterCombinationArray) CDynamicObjectArray<CParameterCombination>;
}

