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

		/// reset to clean state
		virtual void reset(SEXP prhs);

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual int32_t get_int();
		virtual float64_t get_real();
		virtual bool get_bool();

		virtual char* get_string(int32_t& len);

		virtual void get_byte_vector(uint8_t*& vec, int32_t& len);
		virtual void get_char_vector(char*& vec, int32_t& len);
		virtual void get_int_vector(int32_t*& vec, int32_t& len);
		virtual void get_shortreal_vector(float32_t*& vec, int32_t& len);
		virtual void get_real_vector(float64_t*& vec, int32_t& len);
		virtual void get_short_vector(int16_t*& vec, int32_t& len);
		virtual void get_word_vector(uint16_t*& vec, int32_t& len);

		virtual void get_byte_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_char_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_int_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_shortreal_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_real_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_short_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_word_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_byte_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_char_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_int_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_shortreal_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_real_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_short_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_word_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims);

		virtual void get_real_sparsematrix(
			TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_byte_string_list(
			T_STRING<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_char_string_list(
			T_STRING<char>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_int_string_list(
			T_STRING<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_short_string_list(
			T_STRING<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
		virtual void get_word_string_list(
			T_STRING<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);

		/** set functions - to pass data from shogun to the target interface */
		virtual void set_int(int32_t scalar);
		virtual void set_real(float64_t scalar);
		virtual void set_bool(bool scalar);

		virtual bool create_return_values(int32_t num_val);
		virtual void set_byte_vector(const uint8_t* vec, int32_t len);
		virtual void set_char_vector(const char* vec, int32_t len);
		virtual void set_int_vector(const int32_t* vec, int32_t len);
		virtual void set_shortreal_vector(const float32_t* vec, int32_t len);
		virtual void set_real_vector(const float64_t* vec, int32_t len);
		virtual void set_short_vector(const int16_t* vec, int32_t len);
		virtual void set_word_vector(const uint16_t* vec, int32_t len);

		virtual void set_byte_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_char_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_int_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_shortreal_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_real_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_short_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_word_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec);

		virtual void set_real_sparsematrix(
			const TSparse<float64_t>* matrix, int32_t num_feat,
			int32_t num_vec, int64_t nnz);

		virtual void set_byte_string_list(
			const T_STRING<uint8_t>* strings, int32_t num_str);
		virtual void set_char_string_list(
			const T_STRING<char>* strings, int32_t num_str);
		virtual void set_int_string_list(
			const T_STRING<int32_t>* strings, int32_t num_str);
		virtual void set_short_string_list(
			const T_STRING<int16_t>* strings, int32_t num_str);
		virtual void set_word_string_list(
			const T_STRING<uint16_t>* strings, int32_t num_str);

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
