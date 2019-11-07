/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben
 */

%include "GetterVisitorInterface.i"
%{
	namespace shogun
	{

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

		template <typename T>
		struct sg_to_npy_type {};
		SG_TO_NUMPY_TYPE_STRUCT(bool,          NPY_BOOL)
		SG_TO_NUMPY_TYPE_STRUCT(std::vector<bool>::reference,          NPY_BOOL)
		SG_TO_NUMPY_TYPE_STRUCT(char,          NPY_UNICODE)
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
		SG_TO_NUMPY_TYPE_STRUCT(SGObject*,    NPY_OBJECT)
		SG_TO_NUMPY_TYPE_STRUCT(std::shared_ptr<SGObject>,    NPY_OBJECT)
		#undef SG_TO_NUMPY_TYPE_STRUCT

		class PythonVisitor: public GetterVisitorInterface<PythonVisitor, PyObject*>
		{
			friend class GetterVisitorInterface<PythonVisitor, PyObject*>;

		public:
			PythonVisitor(PyObject*& obj): GetterVisitorInterface(obj) {}

		protected:
			template <typename T>
			PyObject* create_array(const T* val, const std::vector<std::ptrdiff_t>& dims)
			{
				T* copy;
				if (dims.size() == 1)
				{
					copy = SG_MALLOC(T, dims[0]);
					std::copy_n(const_cast<T*>(val), dims[0], copy);
				}
				else if (dims.size() == 2)
				{
					copy = SG_MALLOC(T, dims[0] * dims[1]);
					std::copy_n(const_cast<T*>(val), dims[0]*dims[1], copy);
				}
				else
					error("Expected an array with one or two dimensions, but got {}.", dims.size());
				PyArray_Descr* descr=PyArray_DescrFromType(sg_to_npy_type<T>::type);
				PyObject* result = PyArray_NewFromDescr(&PyArray_Type,
					descr, dims.size(), const_cast<npy_intp*>(dims.data()), nullptr, (void*)copy,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, nullptr);
				PyArray_ENABLEFLAGS((PyArrayObject*) result, NPY_ARRAY_OWNDATA);
				return result;
			}

			template <typename>
			PyObject* create_new_list(size_t size)
			{
				// in python we actually use a list, so we don't care about the type
				return PyList_New(0);
			}

			template <typename>
			void append_to_list(PyObject* list, PyObject* v, size_t i)
			{
				PyList_Append(list, v);
			}

			template <typename T>
			PyObject* sg_to_interface(const T* v)
			{
		        // table of conversions from C++ to Python
				if constexpr(std::is_same_v<T, bool>)
					return PyBool_FromLong(*v ? 1 : 0);
				if constexpr(std::is_same_v<T, std::vector<bool>::reference>)
					return PyBool_FromLong(*v ? 1 : 0);
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
				if constexpr(std::is_same_v<T, SGObject*>)
					return SWIG_Python_NewPointerObj(nullptr, SWIG_as_voidptr(*v), SWIGTYPE_p_shogun__SGObject, 0);
				if constexpr(std::is_same_v<T, std::shared_ptr<SGObject>>)
				{
					std::shared_ptr<shogun::SGObject> *smartresult = v ? new std::shared_ptr<shogun::SGObject>(*v) : 0;
					return SWIG_Python_NewPointerObj(nullptr, SWIG_as_voidptr(smartresult), SWIGTYPE_p_std__shared_ptrT_shogun__SGObject_t, SWIG_POINTER_OWN);
				}
				error("Cannot handle casting from shogun type {} to python type!", demangled_type<T>().c_str());
			}
		};
	}
%}
