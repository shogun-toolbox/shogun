#ifndef __MATLABINTERFACE__H_
#define __MATLABINTERFACE__H_

#include <mex.h>

//use compatibility mode w/ matlab <7.x
#if !defined(MX_API_VER) || MX_API_VER<0x07040000
#define mwSize int32_t
#define mwIndex int32_t

#define mxIsLogicalScalar(x) false
#define mxIsLogicalScalarTrue(x) false
#endif

#include <shogun/ui/SGInterface.h>
#include <shogun/lib/config.h>
#include <shogun/features/StringFeatures.h>

using namespace shogun;

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

		virtual void get_vector(uint8_t*& vector, int32_t& len);
		virtual void get_vector(char*& vector, int32_t& len);
		virtual void get_vector(int32_t*& vector, int32_t& len);
		virtual void get_vector(float32_t*& vector, int32_t& len);
		virtual void get_vector(float64_t*& vector, int32_t& len);
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

		virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);

		/*  future versions might support types other than float64_t

		virtual void get_sparse_matrix(SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_sparse_matrix(SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_sparse_matrix(SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_sparse_matrix(SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_sparse_matrix(SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_sparse_matrix(SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);*/

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

		virtual void set_vector(const uint8_t* vec, int32_t len);
		virtual void set_vector(const char* vec, int32_t len);
		virtual void set_vector(const int32_t* vec, int32_t len);
		virtual void set_vector(const float32_t* vec, int32_t len);
		virtual void set_vector(const float64_t* vec, int32_t len);
		virtual void set_vector(const int16_t* vec, int32_t len);
		virtual void set_vector(const uint16_t* vec, int32_t len);

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

		virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat,
			int32_t num_vec, int64_t nnz);

		/*  future versions might support types other than float64_t

		virtual void set_sparse_matrix(const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_sparse_matrix(const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_sparse_matrix(const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_sparse_matrix(const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_sparse_matrix(const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_sparse_matrix(const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec);*/

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

		virtual bool create_return_values(int32_t num)
		{
			return m_nlhs==num;
		}

		/** @return object name */
		inline virtual const char* get_name() const { return "MatlabInterface"; }

		virtual bool cmd_run_python();
		virtual bool cmd_run_octave();
		virtual bool cmd_run_r();

	private:
		const mxArray* get_arg_increment();
		void set_arg_increment(mxArray* arg);

	private:
		mxArray** m_lhs;
		const mxArray** m_rhs;

};
#endif // __MATLABINTERFACE__H_
