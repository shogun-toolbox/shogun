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

	#include <shogun/lib/any.h>
	#include <shogun/io/SGIO.h>
	#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
	extern "C" {
		#include <numpy/arrayobject.h>
	}
	#define SG_TO_NUMPY_TYPE_STRUCT(SG_TYPE, NPY_TYPE) \
	template <>                                        \
	struct sg_to_npy_type<SG_TYPE>                     \
	{                                                  \
		const static NPY_TYPES type = NPY_TYPE;        \
	};

	namespace shogun
	{
		template <typename T>
		struct sg_to_npy_type {};
		SG_TO_NUMPY_TYPE_STRUCT(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
		SG_TO_NUMPY_TYPE_STRUCT(char,          NPY_UNICODE)
#else
		SG_TO_NUMPY_TYPE_STRUCT(char,          NPY_STRING)
#endif
		SG_TO_NUMPY_TYPE_STRUCT(int8_t,        NPY_INT8)
		SG_TO_NUMPY_TYPE_STRUCT(uint8_t,       NPY_UINT8)
		SG_TO_NUMPY_TYPE_STRUCT(int16_t,       NPY_INT16)
		SG_TO_NUMPY_TYPE_STRUCT(uint16_t,      NPY_UINT16)
		SG_TO_NUMPY_TYPE_STRUCT(int32_t,       NPY_INT32)
		SG_TO_NUMPY_TYPE_STRUCT(uint32_t,      NPY_UINT32)
		SG_TO_NUMPY_TYPE_STRUCT(int64_t,       NPY_INT64)
		SG_TO_NUMPY_TYPE_STRUCT(uint64_t,      NPY_UINT64)
		SG_TO_NUMPY_TYPE_STRUCT(float32_t,     NPY_FLOAT32)
		SG_TO_NUMPY_TYPE_STRUCT(float64_t,     NPY_FLOAT64)
		SG_TO_NUMPY_TYPE_STRUCT(complex128_t,  NPY_CDOUBLE)
		SG_TO_NUMPY_TYPE_STRUCT(floatmax_t,    NPY_LONGDOUBLE)
		SG_TO_NUMPY_TYPE_STRUCT(PyObject*,     NPY_OBJECT)
		SG_TO_NUMPY_TYPE_STRUCT(CSGObject*,    NPY_OBJECT)
#undef SG_TO_NUMPY_TYPE_STRUCT

		class PythonVisitor : public AnyVisitor {

		public:
			PythonVisitor(PyObject*& obj) : AnyVisitor(), m_py_obj(&obj) {}

            ~PythonVisitor()
            {
                if (dims)
                    delete[] dims;
            }

			void on(bool *v) final
			{
				handle_sg(v);
			}

			void on(int8_t *v) final
			{
				handle_sg(v);
			}

			void on(int16_t *v) final
			{
				handle_sg(v);
			}

			void on(int32_t *v) final
			{
				handle_sg(v);
			}

			void on(int64_t *v) final
			{
				handle_sg(v);
			}

			void on(float32_t *v) final
			{
				handle_sg(v);
			}

			void on(float64_t *v) final
			{
				handle_sg(v);
			}

			void on(floatmax_t *v) final
			{
				handle_sg(v);
			}

			void on(std::string *v) final
			{
				handle_sg(v->c_str());
			}

			void on(CSGObject **v) final
			{
				handle_sg(v);
			}

			void on(char *v) final
			{
				handle_sg(v);
			}

			void on(uint8_t *v) final
			{
				handle_sg(v);
			}

			void on(uint16_t *v) final
			{
				handle_sg(v);
			}

			void on(uint32_t *v) final
			{
				handle_sg(v);
			}

			void on(uint64_t *v) final
			{
				handle_sg(v);
			}

			void on(complex128_t *v) final
			{
				handle_sg(v);
			}

			void enter_matrix(index_t *rows, index_t *cols) final
			{
				// initialise some variables needed to initialise a pyarray	
				dims = new npy_intp[2];
				dims[0] = (npy_intp) *rows;
				dims[1] = (npy_intp) *cols;
				current_i = 0;
				n_dims = 2;
				nested_current_i = 0;
				m_nested_py_obj = nullptr;
			}

			void enter_vector(index_t *size) final
			{
				dims = new npy_intp[1];
				*dims = (npy_intp) *size;
				current_i = 0;
				n_dims = 1;
				nested_current_i = 0;
				m_nested_py_obj = nullptr;
			}

			void enter_std_vector(size_t *size) final
			{
				*m_py_obj = PyList_New(0);
				current_i = 0;
			}

			void enter_map(size_t *size) final
			{
			}

			void exit_matrix(index_t *rows, index_t *cols) final
			{
				m_nested_py_obj = nullptr;
			}

			void exit_vector(index_t *size) final
			{
				m_nested_py_obj = nullptr;
			}

			void exit_std_vector(size_t *size) final
			{
			}

			void exit_map(size_t *size) final
			{
			}

			void enter_matrix_row(index_t *rows, index_t *cols) final
			{
				current_i = (*rows) * (*cols);
			}

			void exit_matrix_row(index_t *rows, index_t *cols) final
			{
			}

		private:

			PythonVisitor(PyObject*& py_obj, npy_intp* dims_, 
				npy_intp current_i_, int n_dims_): m_py_obj(&py_obj), 
												   dims(dims_), 
												   current_i(current_i_), 
												   n_dims(n_dims_) 
			{
			}

			template <typename T>
			void handle_sg(const T* v)
			{
                // decide how to handle the current value being visited
                // if we created an array dims and this is not a list it must be an array
                // i.e. a value in a SGVector or SGMatrix
				if ((dims && !(*m_py_obj)) || (*m_py_obj && !PyList_Check(*m_py_obj)))
					handle_pyarray(v);
                // this must be a value in std::vector, which will be represented by a pylist
				else if (*m_py_obj && PyList_Check(*m_py_obj))
					handle_pylist(v);
				// this is a scalar, so try to translate it
                else
				  *m_py_obj = sg_to_python(v);
			}

			template <typename T>
			PyObject* sg_to_python(const T* v)
			{
                // table of conversions from C++ to Python
				if constexpr(std::is_same_v<T, bool>)
				{
					PyObject* result = *v ? Py_True : Py_False;
					Py_INCREF(result);
					return result;
				}
				if constexpr(std::is_same_v<T, int8_t>)
					return PyLong_FromLong(static_cast<long>(*v));
				if constexpr(std::is_same_v<T, int16_t>)
					return PyLong_FromLong(static_cast<long>(*v));
				if constexpr(std::is_same_v<T, int32_t>)
					return PyLong_FromLong(*v);
				if constexpr(std::is_same_v<T, int64_t>)
					return PyLong_FromLongLong(*v);
				if constexpr(std::is_same_v<T, float32_t>)
					return PyFloat_FromDouble(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, float64_t>)
					return PyFloat_FromDouble(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, floatmax_t>)
					return PyFloat_FromDouble(static_cast<double>(*v));
				if constexpr(std::is_same_v<T, char>)
					return SWIG_FromCharPtr(v);
				if constexpr(std::is_same_v<T, uint8_t>)
					return PyLong_FromUnsignedLong(static_cast<unsigned long>(*v));
				if constexpr(std::is_same_v<T, uint16_t>)
					return  PyLong_FromUnsignedLong(static_cast<unsigned long>(*v));
				if constexpr(std::is_same_v<T, uint32_t>)
					return PyLong_FromSize_t(static_cast<size_t>(*v));
				if constexpr(std::is_same_v<T, uint64_t>)
					return PyLong_FromUnsignedLong(static_cast<size_t>(*v));
				if constexpr(std::is_same_v<T, complex128_t>)
					return PyComplex_FromDoubles(v->real(), v->imag());
				if constexpr(std::is_same_v<T, CSGObject*>)
					return SWIG_InternalNewPointerObj(SWIG_as_voidptr(*v), SWIGTYPE_p_shogun__CSGObject, 0);
				error("Cannot handle casting from shogun type {} to python type!", demangled_type<T>().c_str());
			}

			template <typename T>
			void handle_pyarray(const T* v)
			{
				if constexpr(std::is_same_v<T, char>)
				{
                    // if it is char we will just get the whole buffer from the SGVector<char>
                    // and make it a string. This is a special case of SGVector, where don't 
                    // convert to a numpy array, but use pylist instead.
                    // Use SWIG because it can handle this depending on Python version. 
					if (!(*m_py_obj))
						*m_py_obj = SWIG_FromCharPtr(v);
				}
				else
				{
                    // this is an array but we haven't instatiated one yet,
                    // so let's do that first
					if (dims && !(*m_py_obj))
					{
						PyArray_Descr* descr=PyArray_DescrFromType(sg_to_npy_type<T>::type);
						*m_py_obj = PyArray_NewFromDescr(&PyArray_Type,
							descr, n_dims, dims, NULL, NULL,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
						PyArray_ENABLEFLAGS((PyArrayObject*) m_py_obj, NPY_ARRAY_OWNDATA);
					}
                    // shouldn't happen
					else if (!dims && !(*m_py_obj))
					{
						return;
					}
                    // assign value v to the ith element of array buffer
                    // which is in column major
					((T*)PyArray_DATA((PyArrayObject*)*m_py_obj))[current_i] = *v;
				}
				current_i++;
			}

			template <typename T>
			void handle_pylist(const T* v)
			{
				bool new_obj = m_nested_py_obj ? false : true;
				auto nested_visitor = PythonVisitor(m_nested_py_obj, dims, nested_current_i, n_dims);
				nested_visitor.on(const_cast<T*>(v));
				if (m_nested_py_obj && new_obj)
				{
				  PyList_Append(*m_py_obj, m_nested_py_obj);
				  current_i++;
				}
				else if (!m_nested_py_obj && !new_obj)
					error("Could not cast shogun type {} to python type!", demangled_type<T>().c_str());

				nested_current_i++;
			}

			PyObject** m_py_obj;
			npy_intp* dims = nullptr;
			npy_intp current_i;
			int n_dims;
			PyObject* m_nested_py_obj = nullptr;
			npy_intp nested_current_i;
		};
	}
%}

namespace shogun {
	%extend CSGObject
	{
		PyObject* get(const std::string& name) const
		{
			PyObject* result = nullptr;
			try
			{
				const auto params = $self->get_params();
				auto find_iter = params.find(name);

				if (find_iter == params.end())
				{
					error("Could not find parameter {} in {}.", name.c_str(), $self->get_name());
					return nullptr;
				}

				auto visitor = PythonVisitor(result);
				find_iter->second->get_value().visit(&visitor);
				if (!result)
                {
					error("Could not cast parameter {} to python object!", name.c_str());
					return nullptr;
			    }
            }
			catch(ShogunException& e)
			{
				SWIG_Error(SWIG_SystemError, const_cast<char*>(e.what()));
				return nullptr;
			}
			return result;
		}
		PyObject* get(const std::string& name, int index) const
		{
			PyObject* result = nullptr;
			try
			{
				CSGObject* obj = $self->get(name, index);
				result = SWIG_InternalNewPointerObj(SWIG_as_voidptr(obj), SWIGTYPE_p_shogun__CSGObject, 0);
			}
			catch(ShogunException& e)
			{
				SWIG_Error(SWIG_SystemError, const_cast<char*>(e.what()));
				return nullptr;
			}
			return result;
		}
	}
}

%ignore get;

%feature("python:slot", "tp_str", functype="reprfunc") shogun::CSGObject::__str__;
%feature("python:slot", "tp_repr", functype="reprfunc") shogun::CSGObject::__repr__;
/*%feature("python:slot", "tp_hash", functype="hashfunc") shogun::CSGObject::myHashFunc;*/
%feature("python:tp_print") shogun::CSGObject "print_sgobject";
/*%feature("python:slot", "tp_as_buffer", functype="PyBufferProcs*") shogun::SGObject::tp_as_buffer;
%feature("python:slot", "bf_getbuffer", functype="getbufferproc") shogun::SGObject::getbuffer;*/
#endif // SWIGPYTHON

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "shogun_doxygen.i"
#endif
#endif

%include "std_vector.i"
%include "shogun_ignores.i"
%include "RandomMixin.i"
%include "Machine_includes.i"
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
%include "IO.i"
%include "Library.i"
%include "Mathematics.i"
%include "Features.i"
%include "Transformer.i"
%include "Converter.i"
%include "Preprocessor.i"
%include "Machine.i"
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

%define PUT_ENUM_INT_DISPATCHER(TAG, VALUE)
    auto string_to_enum_map = $self->get_string_to_enum_map();
    if (string_to_enum_map.find(TAG.name()) == string_to_enum_map.end()) {
        $self->put(TAG, VALUE);
        return;
    }
    auto val = static_cast<machine_int_t>(VALUE);
    auto string_to_enum = string_to_enum_map[TAG.name()];
    auto count = std::count_if(string_to_enum.begin(), string_to_enum.end(),
                               [val](const std::pair <std::string, machine_int_t> &p) {
                                   return val == p.second;
                               });
    if (count > 0)
    {
        $self->put(Tag<machine_int_t>(TAG.name()), val);
    }
    else
    {
        error("There is no option in {}::{} for value {}",
                $self->get_name(), TAG.name().c_str(), val);
    }
%enddef

%define PUT_SCALAR_DISPATCHER(Type, name, value)
	Tag<Type> tag_t(name);
	Tag<int32_t> tag_int32(name);
	Tag<int64_t> tag_int64(name);
	Tag<float64_t> tag_float64(name);

	if ($self->has(tag_int32))
	{
		PUT_ENUM_INT_DISPATCHER(tag_int32, (int32_t) value);
	}
	else if ($self->has(tag_int64))
	{
		PUT_ENUM_INT_DISPATCHER(tag_int64, (int64_t) value);
	}
	else if ($self->has(tag_float64))
		$self->put(tag_float64, (float64_t)value);
#ifdef SWIGR
	else if (Tag<SGVector<Type>> tag_tvec(name); $self->has(tag_tvec))
	{
		SGVector<Type> vec(1);
		vec[0] = value;
		$self->put(tag_tvec, vec);
	}
	else if (Tag<SGVector<int32_t>> tag_int32vec(name); $self->has(tag_int32vec))
	{
		SGVector<int32_t> vec(1);
		vec[0] = value;
		$self->put(tag_int32vec, vec);
	}
	else if (Tag<SGVector<float64_t>> tag_float64vec(name); $self->has(tag_float64vec))
	{
		SGVector<float64_t> vec(1);
		vec[0] = value;
		$self->put(tag_float64vec, vec);
	}
	else if (Tag<SGVector<bool>> tag_boolvec(name); $self->has(tag_boolvec))
	{
		SGVector<bool> vec(1);
		vec[0] = value;
		$self->put(tag_boolvec, vec);
	}
#endif
	else
		$self->put(tag_t, value);
%enddef

namespace shogun
{
%extend CSGObject
{
	template <typename T, typename U = typename std::enable_if_t<std::is_arithmetic<T>::value>>
	void put_scalar_dispatcher(const std::string& name, T value)
	{
		PUT_SCALAR_DISPATCHER(T, name, value)
	}

	/* get method for strings to disambiguate it from get_option */
	std::string get_string(const std::string& name) const
	{
		return $self->get<std::string>(name);
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
				SGVector<X> vec(mat);
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

#ifdef SWIGR
	template <typename T, typename X = typename std::enable_if_t<std::is_same<SGVector<typename extract_value_type<T>::value_type>, T>::value> >
	void put_vector_scalar_dispatcher(const std::string& name, T vector)
	{
		if (Tag<T> tag_vec(name); vector.size() > 1 || $self->has(tag_vec))
		{
			$self->put(tag_vec, vector);
		}
		else
		{
			auto value = vector[0];
			PUT_SCALAR_DISPATCHER(X, name, value)
		}
	}
#endif // SWIGR
}

%template(put) CSGObject::put_scalar_dispatcher<int32_t, int32_t>;
#ifndef SWIGJAVA
%template(put) CSGObject::put_scalar_dispatcher<int64_t, int64_t>;
#endif // SWIGJAVA
%template(put) CSGObject::put_scalar_dispatcher<float64_t, float64_t>;
%template(put) CSGObject::put_scalar_dispatcher<bool, bool>;

#ifndef SWIGR
%template(put) CSGObject::put<SGVector<bool>, SGVector<bool>>;
#endif // SWIGR

#if !defined(SWIGJAVA) && !defined(SWIGR)
%template(put) CSGObject::put<SGVector<int32_t>, SGVector<int32_t>>;
%template(put) CSGObject::put<SGVector<float64_t>, SGVector<float64_t>>;
#elif defined(SWIGJAVA)
%template(put) CSGObject::put_vector_or_matrix_from_double_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
#elif defined(SWIGR)
%template(put) CSGObject::put_vector_scalar_dispatcher<SGVector<bool>, bool>;
%template(put) CSGObject::put_vector_scalar_dispatcher<SGVector<int32_t>, int32_t>;
%template(put) CSGObject::put_vector_scalar_dispatcher<SGVector<float64_t>, float64_t>;
#endif

#ifndef SWIGJAVA
%template(put) CSGObject::put<SGMatrix<float64_t>, SGMatrix<float64_t>>;
#endif // SWIGJAVA

#ifndef SWIGPYTHON
%template(get_real) CSGObject::get<float64_t, void>;
%template(get_int) CSGObject::get<int32_t, void>;
%template(get_long) CSGObject::get<int64_t, void>;
%template(get_real_matrix) CSGObject::get<SGMatrix<float64_t>, void>;
%template(get_char_string_list) CSGObject::get<std::vector<SGVector<char>>, void>;
%template(get_word_string_list) CSGObject::get<std::vector<SGVector<uint16_t>>, void>;
%template(get_option) CSGObject::get<std::string, void>;

#ifndef SWIGJAVA
%template(get_real_vector) CSGObject::get<SGVector<float64_t>, void>;
%template(get_int_vector) CSGObject::get<SGVector<int32_t>, void>;
#else // SWIGJAVA
%template(get_real_vector) CSGObject::get_vector_as_matrix_dispatcher<SGMatrix<float64_t>, float64_t>;
%template(get_int_vector) CSGObject::get_vector_as_matrix_dispatcher<SGMatrix<int32_t>, int32_t>;
#endif // SWIGJAVA
#endif // SWIGPYTHON
%template(put) CSGObject::put<std::string, std::string>;

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
PUT_ADD(CInference)
PUT_ADD(CDifferentiableFunction)
PUT_ADD(CNeuralLayer)
PUT_ADD(CSplittingStrategy)
PUT_ADD(CEvaluation)
PUT_ADD(CSVM)
PUT_ADD(CMeanFunction)
PUT_ADD(CLikelihoodModel)
PUT_ADD(CTokenizer)

%template(kernel) kernel<float64_t, float64_t>;
%template(features) features<float64_t>;


} // namespace shogun


