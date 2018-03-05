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
%include "factory.i"

#if defined(SWIGPERL)
%include "abstract_types_extension.i"
#endif

%pragma(java) moduleimports=%{
import org.jblas.*;
%}

namespace shogun
{
%extend CSGObject
{
	template <typename T, typename U= typename std::enable_if_t<std::is_arithmetic<T>::value>>
	void put_scalar_dispatcher(const std::string& name, T value)
	{
		Tag<T> tag_t(name);
		Tag<int32_t> tag_int32(name);
		Tag<int64_t> tag_int64(name);
		Tag<float64_t> tag_float64(name);

		if ($self->has(tag_int32))
			$self->put(tag_int32, (int32_t)value);
		else if ($self->has(tag_int64))
			$self->put(tag_int64, (int64_t)value);
		else if ($self->has(tag_float64))
			$self->put(tag_float64, (float64_t)value);
		else
			$self->put(tag_t, value);
	}
	
#ifdef SWIGJAVA
	template <typename T, typename X = typename std::enable_if_t<std::is_same<SGMatrix<typename extract_value_type<T>::value_type>, T>::value> >
	void put_vector_or_matrix_dispatcher(const std::string& name, T value)
	{
		Tag<SGVector<X>> tag_vec(name);
		Tag<T> tag_mat(name);
	
		if ((value.num_rows==1 || value.num_cols==1) && $self->has(tag_vec))
		{
			SGVector<X> vec(value.data(), value.size(), false);
			$self->put(tag_vec, vec);
		}
		else
			$self->put(tag_mat, value);
	}
	
	template <typename T, typename X = typename std::enable_if_t<std::is_same<SGMatrix<typename extract_value_type<T>::value_type>, T>::value> >
	T get_vector_as_matrix_dispatcher(const std::string& name)
	{
		SGVector<X> vec = $self->get<SGVector<X>>(name);
		T mat(vec.data(), 1, vec.vlen, false);
		return mat;
	}
#endif // SWIGJAVA
}

%template(put) CSGObject::put_scalar_dispatcher<int32_t, int32_t>;
#ifndef SWIGJAVA
%template(put) CSGObject::put_scalar_dispatcher<int64_t, int64_t>;
#endif // SWIGJAVA
%template(put) CSGObject::put_scalar_dispatcher<float64_t, float64_t>;


#ifndef SWIGJAVA
%template(put) CSGObject::put<SGVector<float64_t>, SGVector<float64_t>>;
%template(put) CSGObject::put<SGMatrix<float64_t>, SGMatrix<float64_t>>;
#else // SWIGJAVA
%template(put) CSGObject::put_vector_or_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
#endif // SWIGJAVA

%template(get_real) CSGObject::get<float64_t, void>;
%template(get_real_matrix) CSGObject::get<SGMatrix<float64_t>, void>;
#ifndef SWIGJAVA
%template(get_real_vector) CSGObject::get<SGVector<float64_t>, void>;
#else // SWIGJAVA
%template(get_real_vector) CSGObject::get_vector_as_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
#endif // SWIGJAVA

} // namespace shogun
