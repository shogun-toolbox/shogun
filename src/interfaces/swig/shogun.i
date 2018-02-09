/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Soeren Sonnenburg
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

/* This is needed with SWIG >= 3.0.5 on MacOSX because of
   conflicting macros in AssertMacros.h of Carbon-framework.
   Won't cause harm on other distros. */
%{
#if defined(__APPLE__)
#define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#endif // defined(__APPLE__)
%}

%include "swig_config.h"

%define DOCSTR
"The `shogun` module gathers all modules available in the SHOGUN toolkit."
%enddef

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%module(directors="1", docstring=DOCSTR) shogun
#else
%module(docstring=DOCSTR) shogun
#endif
#undef DOCSTR


/* Documentation */
%feature("autodoc","0");

#ifdef SWIGPYTHON
#include <object.h>
%{
    static int print_sgobject(PyObject *pyobj, FILE *f, int flags);
%}

%feature("python:slot", "tp_str", functype="reprfunc") shogun::CSGObject::__str__;
%feature("python:slot", "tp_repr", functype="reprfunc") shogun::CSGObject::__repr__;
/*%feature("python:slot", "tp_hash", functype="hashfunc") shogun::CSGObject::myHashFunc;*/
%feature("python:tp_print") shogun::CSGObject "print_sgobject";
/*%feature("python:slot", "tp_as_buffer", functype="PyBufferProcs*") shogun::SGObject::tp_as_buffer;
%feature("python:slot", "bf_getbuffer", functype="getbufferproc") shogun::SGObject::getbuffer;*/
#endif

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "shogun_doxygen.i"
#endif
#endif

%include "shogun_ignores.i"

%include "Classifier_includes.i"
%include "Clustering_includes.i"
%include "Distance_includes.i"
%include "Distribution_includes.i"
%include "Evaluation_includes.i"
%include "Features_includes.i"
%include "IO_includes.i"
%include "Kernel_includes.i"
%include "Library_includes.i"
%include "Mathematics_includes.i"
%include "Converter_includes.i"
%include "Preprocessor_includes.i"
%include "Regression_includes.i"
%include "Structure_includes.i"
%include "Multiclass_includes.i"
%include "Transfer_includes.i"
%include "Loss_includes.i"
%include "Statistics_includes.i"
%include "Latent_includes.i"
%include "Metric_includes.i"
%include "Minimizer_includes.i"
%include "GaussianProcess_includes.i"
%include "ModelSelection_includes.i"
%include "Ensemble_includes.i"
%include "NeuralNets_includes.i"
%include "bagging_includes.i"
%include "Boost_includes.i"

%include "SGBase.i"
%include "Machine.i"
%include "IO.i"
%include "Library.i"
%include "Mathematics.i"
%include "Features.i"
%include "Converter.i"
%include "Preprocessor.i"
%include "Evaluation.i"
%include "Distance.i"
%include "Kernel.i"
%include "Distribution.i"
%include "Classifier.i"
%include "Regression.i"
%include "Clustering.i"
%include "Structure.i"
%include "Multiclass.i"
%include "Transfer.i"
%include "Loss.i"
%include "Statistics.i"
%include "Latent.i"
%include "Metric.i"
%include "Minimizer.i"
%include "ModelSelection.i"
%include "GaussianProcess.i"
%include "Ensemble.i"
%include "NeuralNets.i"
%include "bagging.i"
%include "Boost.i"

%include "ParameterObserver.i"


#ifdef SWIGOCTAVE
// Octave treats 4.0 as integer, so need conversion when setting float parameters
%include <shogun/base/SGObject.h>
namespace shogun {
%extend CSGObject {
	void put(const std::string& name, const long value) throw(ShogunException)
	{
		Tag<long> tag_int(name);
		Tag<double> tag_float(name);
		
		// octave treats 4.0 as an integer
		if ($self->has(tag_int))
			$self->put(tag_int, value);
		else if ($self->has(tag_float))
		{
			SG_SWARNING("Converting %d of type int64 to %f of type float64 "
						"when setting parameter %s::%s. Octave treats 4.0 as integer.\n",
				value, (float64_t)value,$self->get_name(), name.c_str());
			$self->put(tag_float, (double) value);
		}
		else
			// to get original error message
			$self->put(tag_int, value);
	}
}
}
#endif

%define SUPPORT_TAG(short_type, type)
    %template(put) shogun::CSGObject::put<type, void>;
    %template(get_ ## short_type) shogun::CSGObject::get<type, void>;
%enddef

SUPPORT_TAG(double, float64_t)
#ifdef SWIGOCTAVE
	// already defined a custom put above
    %template(get_int) shogun::CSGObject::get<int64_t, void>;
#else
	SUPPORT_TAG(int, int64_t)
#endif
#ifndef SWIGJAVA
	// Java treats everything as a matrix
	SUPPORT_TAG(real_vector, SGVector<float64_t>)
#endif
SUPPORT_TAG(real_matrix, SGMatrix<float64_t>)

#if defined(SWIGPERL)
%include "abstract_types_extension.i"
#endif
