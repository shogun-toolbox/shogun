#ifndef __RINTERFACE__H_
#define __RINTERFACE__H_

#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)
#include "interface/SGInterface.h"

#include <Rdefines.h>

class CRInterface : public CSGInterface
{
	public:
		CRInterface(SEXP prhs);
		~CRInterface();

		/** get functions - to pass data from the target interface to shogun */
		virtual void parse_args(INT num_args, INT num_default_args);

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual INT get_int();
		virtual DREAL get_real();
		virtual bool get_bool();

		virtual CHAR* get_string(INT& len);

		virtual void get_byte_vector(BYTE*& vec, INT& len);
		virtual void get_char_vector(CHAR*& vec, INT& len);
		virtual void get_int_vector(INT*& vec, INT& len);
		virtual void get_shortreal_vector(SHORTREAL*& vec, INT& len);
		virtual void get_real_vector(DREAL*& vec, INT& len);
		virtual void get_short_vector(SHORT*& vec, INT& len);
		virtual void get_word_vector(WORD*& vec, INT& len);

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
		virtual bool create_return_values(INT num_val);
		virtual void set_byte_vector(const BYTE* vec, INT len);
		virtual void set_char_vector(const CHAR* vec, INT len);
		virtual void set_int_vector(const INT* vec, INT len);
		virtual void set_shortreal_vector(const SHORTREAL* vec, INT len);
		virtual void set_real_vector(const DREAL* vec, INT len);
		virtual void set_short_vector(const SHORT* vec, INT len);
		virtual void set_word_vector(const WORD* vec, INT len);

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

		SEXP get_return_values();

	private:
		const SEXP get_arg_increment()
		{
			ASSERT(m_rhs_counter>=0 && m_rhs_counter<m_nrhs+1); // +1 for action
			m_rhs=CDR(m_rhs);
			m_rhs_counter++;

			return m_rhs;
		}

		void set_arg_increment(SEXP arg)
		{
			ASSERT(m_lhs_counter>=0 && m_lhs_counter<m_nlhs);
			SET_VECTOR_ELT(m_lhs, m_lhs_counter, arg);
			m_lhs_counter++;
		}

	private:
		SEXP m_lhs;
		SEXP m_rhs;
};
#endif // HAVE_R && !HAVE_SWIG
#endif // __RINTERFACE__H_
