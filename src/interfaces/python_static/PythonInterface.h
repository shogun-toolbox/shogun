#ifndef __PYTHONINTERFACE__H_
#define __PYTHONINTERFACE__H_

#undef _POSIX_C_SOURCE
#undef HAVE_STAT

extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

#include <ui/SGInterface.h>

extern "C" {
#include <numpy/arrayobject.h>
}

#if PY_VERSION_HEX >= 0x03000000
    #define IS_PYTHON3

    #define PyInt_Check PyLong_Check
    #define PyInt_AS_LONG PyLong_AS_LONG
#endif

// Python2/3 module initialization
#ifdef IS_PYTHON3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, methods) \
        static struct PyModuleDef moduledef = { \
            PyModuleDef_HEAD_INIT, name, NULL, -1, methods, }; \
            ob = PyModule_Create(&moduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) PyMODINIT_FUNC init##name(void)
    #define MOD_DEF(ob, name, methods) \
          ob = Py_InitModule(name, methods);
#endif

#ifdef IS_PYTHON3
int init_numpy()
{
	import_array();
}
#else
void init_numpy()
{
	import_array();
}
#endif

namespace shogun
{
class CPythonInterface : public CSGInterface
{
	public:
		CPythonInterface(PyObject* args);
		CPythonInterface(PyObject* self, PyObject* args);
		~CPythonInterface();

		/// reset to clean state
		virtual void reset(PyObject* self, PyObject* args);

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual int32_t get_int();
		virtual float64_t get_real();
		virtual bool get_bool();

		virtual char* get_string(int32_t& len);

		virtual void get_vector(uint8_t*& vector, int32_t& len);
		virtual void get_vector(char*& vector, int32_t& len);
		virtual void get_vector(int32_t*& vector, int32_t& len);
		virtual void get_vector(float64_t*& vector, int32_t& len);
		virtual void get_vector(float32_t*& vector, int32_t& len);
		virtual void get_vector(int16_t*& vector, int32_t& len);
		virtual void get_vector(uint16_t*& vector, int32_t& len);

		virtual void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_string_list(
			SGString<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);

		virtual void get_attribute_struct(
			const CDynamicArray<T_ATTRIBUTE>* &attrs);

		/** set functions - to pass data from shogun to the target interface */
		virtual void set_int(int32_t scalar);
		virtual void set_real(float64_t scalar);
		virtual void set_bool(bool scalar);

		virtual void set_vector(const uint8_t* vector, int32_t len);
		virtual void set_vector(const char* vector, int32_t len);
		virtual void set_vector(const int32_t* vector, int32_t len);
		virtual void set_vector(
			const float32_t* vector, int32_t len);
		virtual void set_vector(const float64_t* vector, int32_t len);
		virtual void set_vector(const int16_t* vector, int32_t len);
		virtual void set_vector(const uint16_t* vector, int32_t len);

		virtual void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec);

		virtual void get_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims);

		virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat,
			int32_t num_vec, int64_t nnz);

		virtual void set_string_list(
			const SGString<uint8_t>* strings, int32_t num_str);
		virtual void set_string_list(
			const SGString<char>* strings, int32_t num_str);
		virtual void set_string_list(
			const SGString<int32_t>* strings, int32_t num_str);
		virtual void set_string_list(
			const SGString<int16_t>* strings, int32_t num_str);
		virtual void set_string_list(
			const SGString<uint16_t>* strings, int32_t num_str);

		virtual void set_attribute_struct(
			const CDynamicArray<T_ATTRIBUTE>* attrs);

		virtual bool create_return_values(int32_t num);

		PyObject* get_return_values()
		{
            if (m_nlhs==1)
            {
                PyObject* retval=PyTuple_GET_ITEM(m_lhs, 0);
                Py_INCREF(retval);
                Py_DECREF(m_lhs);
                m_lhs=retval;
            }
			return m_lhs;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "PythonInterface"; }

		static void run_python_init();
		static void run_python_exit();
		static bool run_python_helper(CSGInterface* from_if);
		virtual bool cmd_run_octave();
		virtual bool cmd_run_r();

	private:
		const PyObject* get_arg_increment()
		{
			const PyObject* retval;
			ASSERT(m_rhs_counter>=0 && m_rhs_counter<m_nrhs+1); // +1 for action
			ASSERT(m_rhs);

			retval=PyTuple_GET_ITEM(m_rhs, m_rhs_counter);
			m_rhs_counter++;

			return retval;
		}

		void set_arg_increment(PyObject* arg)
		{
			ASSERT(m_lhs_counter>=0 && m_lhs_counter<m_nlhs);
			ASSERT(m_lhs);
            //Py_INCREF(arg);
			PyTuple_SET_ITEM(m_lhs, m_lhs_counter, arg);
			m_lhs_counter++;
		}

	private:
		static void* m_pylib;
		PyObject* m_lhs;
		PyObject* m_rhs;
};
}
#endif // __PYTHONINTERFACE__H_
