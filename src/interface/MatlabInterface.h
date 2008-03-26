#ifndef __MATLABINTERFACE__H_
#define __MATLABINTERFACE__H_

#include "lib/config.h"

#if defined(HAVE_MATLAB) && !defined(HAVE_SWIG)

#include "lib/matlab.h"
#include "features/StringFeatures.h"

#include "interface/SGInterface.h"


class CMatlabInterface : public CSGInterface
{
	public:
		CMatlabInterface(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]);
		~CMatlabInterface();

		/** get functions - to pass data from the target interface to shogun */
		virtual void parse_args(INT num_args, INT num_default_args);

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual INT get_int();
		virtual DREAL get_real();
		virtual bool get_bool();

		virtual CHAR* get_string(INT& len);

		virtual void get_byte_vector(BYTE*& vector, INT& len);
		virtual void get_char_vector(CHAR*& vector, INT& len);
		virtual void get_int_vector(INT*& vector, INT& len);
		virtual void get_shortreal_vector(SHORTREAL*& vector, INT& len);
		virtual void get_real_vector(DREAL*& vector, INT& len);
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

		/*  future versions might support types other than DREAL

		virtual void get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec);
		virtual void get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec);*/

		virtual void get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len);
		virtual void get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len);

		/** set functions - to pass data from shogun to the target interface */
		virtual void create_return_values(INT num_val);
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

		virtual void set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec);

		/*  future versions might support types other than DREAL
		
		virtual void set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec);
		virtual void set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec);
		virtual void set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec);
		virtual void set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec);
		virtual void set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec);
		virtual void set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec);*/

		virtual void set_byte_string_list(const T_STRING<BYTE>* strings, INT num_str);
		virtual void set_char_string_list(const T_STRING<CHAR>* strings, INT num_str);
		virtual void set_int_string_list(const T_STRING<INT>* strings, INT num_str);
		virtual void set_short_string_list(const T_STRING<SHORT>* strings, INT num_str);
		virtual void set_word_string_list(const T_STRING<WORD>* strings, INT num_str);

		virtual void submit_return_values();

	private:
		const mxArray* get_arg_increment();
		void set_arg_increment(mxArray* arg);

	private:
		mxArray** m_lhs;
		const mxArray** m_rhs;

};
#endif // HAVE_MATLAB && HAVE_SWIG
#endif // __MATLABINTERFACE__H_
