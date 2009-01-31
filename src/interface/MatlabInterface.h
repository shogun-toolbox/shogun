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
		CMatlabInterface(
			int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);
		~CMatlabInterface();

		/// reset to clean state
		virtual void reset(
			int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]);

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual int32_t get_int();
		virtual float64_t get_real();
		virtual bool get_bool();

		virtual char* get_string(int32_t& len);

		virtual void get_byte_vector(uint8_t*& vector, int32_t& len);
		virtual void get_char_vector(char*& vector, int32_t& len);
		virtual void get_int_vector(int32_t*& vector, int32_t& len);
		virtual void get_shortreal_vector(float32_t*& vector, int32_t& len);
		virtual void get_real_vector(float64_t*& vector, int32_t& len);
		virtual void get_short_vector(int16_t*& vector, int32_t& len);
		virtual void get_word_vector(uint16_t*& vector, int32_t& len);

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

		/*  future versions might support types other than float64_t

		virtual void get_byte_sparsematrix(TSparse<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_char_sparsematrix(TSparse<char>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_int_sparsematrix(TSparse<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_shortreal_sparsematrix(TSparse<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_short_sparsematrix(TSparse<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_word_sparsematrix(TSparse<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);*/

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

		/*  future versions might support types other than float64_t
		
		virtual void set_byte_sparsematrix(const TSparse<uint8_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_char_sparsematrix(const TSparse<char>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_int_sparsematrix(const TSparse<int32_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_shortreal_sparsematrix(const TSparse<float32_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_short_sparsematrix(const TSparse<int16_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_word_sparsematrix(const TSparse<uint16_t>* matrix, int32_t num_feat, int32_t num_vec);*/

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

		virtual bool create_return_values(int32_t num)
		{
			return m_nlhs==num;
		}

		/** @return object name */
		inline virtual const char* get_name() { return "MatlabInterface"; }

	private:
		const mxArray* get_arg_increment();
		void set_arg_increment(mxArray* arg);

	private:
		mxArray** m_lhs;
		const mxArray** m_rhs;

};
#endif // HAVE_MATLAB && HAVE_SWIG
#endif // __MATLABINTERFACE__H_
