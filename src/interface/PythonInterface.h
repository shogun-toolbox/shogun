#ifndef __PYTHONINTERFACE__H_
#define __PYTHONINTERFACE__H_

#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)            
#include "interface/SGInterface.h"

#include "lib/python.h"

extern "C" {
#include <numpy/arrayobject.h>
}

class CPythonInterface : public CSGInterface
{
	public:
		CPythonInterface(PyObject* self, PyObject* args);
		~CPythonInterface();

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual INT get_int();
		virtual DREAL get_real();
		virtual bool get_bool();

		virtual CHAR* get_string(INT& len);

		virtual void get_byte_vector(BYTE*& vector, INT& len);
		virtual void get_char_vector(CHAR*& vector, INT& len);
		virtual void get_int_vector(INT*& vector, INT& len);
		virtual void get_real_vector(DREAL*& vector, INT& len);
		virtual void get_shortreal_vector(SHORTREAL*& vector, INT& len);
		virtual void get_short_vector(SHORT*& vector, INT& len);
		virtual void get_word_vector(WORD*& vector, INT& len);

		virtual void get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec);

		virtual void get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec);

		virtual void get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len);


		/** set functions - to pass data from shogun to the target interface */
		virtual void set_int(INT scalar);
		virtual void set_real(DREAL scalar);
		virtual void set_bool(bool scalar);

		virtual void set_byte_vector(const BYTE* vector, INT len);
		virtual void set_char_vector(const CHAR* vector, INT len);
		virtual void set_int_vector(const INT* vector, INT len);
		virtual void set_shortreal_vector(const SHORTREAL* vector, INT len);
		virtual void set_real_vector(const DREAL* vector, INT len);
		virtual void set_short_vector(const SHORT* vector, INT len);
		virtual void set_word_vector(const WORD* vector, INT len);

		virtual void set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec);
		virtual void set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec);
		virtual void set_int_matrix(const INT* matrix, INT num_feat, INT num_vec);
		virtual void set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec);
		virtual void set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec);
		virtual void set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec);
		virtual void set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec);

		virtual void set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec, LONG nnz);

		virtual void set_byte_string_list(const T_STRING<BYTE>* strings, INT num_str);
		virtual void set_char_string_list(const T_STRING<CHAR>* strings, INT num_str);
		virtual void set_int_string_list(const T_STRING<INT>* strings, INT num_str);
		virtual void set_short_string_list(const T_STRING<SHORT>* strings, INT num_str);
		virtual void set_word_string_list(const T_STRING<WORD>* strings, INT num_str);

		virtual bool create_return_values(INT num);

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
		PyObject* m_lhs;
		PyObject* m_rhs;
};
#endif // HAVE_PYTHON && !HAVE_SWIG
#endif // __PYTHONINTERFACE__H_
