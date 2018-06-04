/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Giovanni De Toni, Sergey Lisitsyn
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
%include "Transformer.i"
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
	// templated since otherwise SWIG doesn't match the typemap for SGMatrix
	// for the DoubleMatrix hack, X = float64_t and T = SGMatrix<X>
	template <typename T, typename X = typename std::enable_if_t<std::is_same<SGMatrix<typename extract_value_type<T>::value_type>, T>::value> >
	void put_vector_or_matrix_from_double_matrix_dispatcher(const std::string& name, T mat)
	{
		Tag<T> tag_input_mat(name);
		Tag<SGVector<X>> tag_vec_X(name);
		Tag<SGVector<int32_t>> tag_vec_int32(name);
		Tag<SGVector<bool>> tag_vec_bool(name);
	
		// simplest case: types are as given
		if ($self->has(tag_input_mat))
		{
			$self->put(tag_input_mat, mat);
			return;
		}
	
		// tag didnt match: either it was vector, or has different inner type
	
		// definitely a matrix, might need to convert values
		if (mat.num_rows>1 && mat.num_cols>1)
		{
			// TODO once needed
		}
		// maybe input was vector
		else
		{
			// vector with correct inner type
			if ($self->has(tag_vec_X))
			{
				SGVector<X> vec(mat.data(), mat.size(), false);
				$self->put(tag_vec_X, vec);
				return;
			}
			// below are vectors which needs to be converted
			else if ($self->has(tag_vec_int32))
			{
				SGVector<int32_t> vec(mat.size());
				std::transform(mat.begin(), mat.end(), vec.begin(),
						[](X e) { return (int32_t)e; });
				$self->put(tag_vec_int32, vec);
				return;
			}
			else if ($self->has(tag_vec_bool))
			{
				SGVector<bool> vec(mat.size());
				std::transform(mat.begin(), mat.end(), vec.begin(),
						[](X e) { return (bool)e; });
				$self->put(tag_vec_bool, vec);
				return;
			}
		}
		
		// final fall-back in case user did a mistake
		$self->put(tag_input_mat, mat);
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
%template(put) CSGObject::put_scalar_dispatcher<bool, bool>;


%template(put) CSGObject::put<SGVector<bool>, SGVector<bool>>;
#ifndef SWIGJAVA
%template(put) CSGObject::put<SGVector<int32_t>, SGVector<int32_t>>;
%template(put) CSGObject::put<SGVector<float64_t>, SGVector<float64_t>>;
%template(put) CSGObject::put<SGMatrix<float64_t>, SGMatrix<float64_t>>;
#else // SWIGJAVA
%template(put) CSGObject::put_vector_or_matrix_from_double_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
#endif // SWIGJAVA

%template(get_real) CSGObject::get<float64_t, void>;
%template(get_int) CSGObject::get<int32_t, void>;
%template(get_real_matrix) CSGObject::get<SGMatrix<float64_t>, void>;
#ifndef SWIGJAVA
%template(get_real_vector) CSGObject::get<SGVector<float64_t>, void>;
%template(get_int_vector) CSGObject::get<SGVector<int32_t>, void>;
#else // SWIGJAVA
%template(get_real_vector) CSGObject::get_vector_as_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
%template(get_int_vector) CSGObject::get_vector_as_matrix_dispatcher<SGMatrix<int32_t>, int32_t>;
#endif // SWIGJAVA

%define PUT_ADD(sg_class)
%template(put) CSGObject::put<sg_class, sg_class, void>;
%template(add) CSGObject::add<sg_class, sg_class>;
%enddef

PUT_ADD(CMachine)
PUT_ADD(CKernel)
PUT_ADD(CDistance)
PUT_ADD(CFeatures)
PUT_ADD(CLabels)
PUT_ADD(CECOCEncoder)
PUT_ADD(CECOCDecoder)
PUT_ADD(CMulticlassStrategy)
PUT_ADD(CCombinationRule)
PUT_ADD(CDifferentiableFunction)

%template(kernel) kernel<float64_t, float64_t>;
%template(features) features<float64_t>;


} // namespace shogun
