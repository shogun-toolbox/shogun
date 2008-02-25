#include "lib/common.h"
#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"

class CGUIInterface
{
	public:
		enum IFType
		{
			UNDEFINED,

			SCALAR_INT,
			SCALAR_REAL,
			SCALAR_BOOL,
			SINGLE_STRING,

			VECTOR_BYTE,
			VECTOR_INT,
			VECTOR_SHORTREAL,
			VECTOR_REAL,

			MATRIX_BYTE,
			MATRIX_INT,
			MATRIX_SHORTREAL,
			MATRIX_REAL,

			STRING_LIST
		};

	public:
		CGUIInterface();
		{
		}

		~CGUIInterface();
		{
		}

		/** get functions - to pass data from the target interface to shogun */
		void parse_args(INT num_args, INT num_default_args)=0;

		/// get type of current argument (does not increment argument counter)
		IFType get_argument_type();

		INT get_int()=0;
		DREAL get_real()=0;
		bool get_bool()=0;

		CHAR* get_string()=0;
		INT get_int_from_string()=0;
		DREAL get_real_from_string()=0;
		bool get_bool_from_string()=0;

		//get_byte_vector(BYTE*& bytes, INT& len);
		void get_byte_vector(BYTE** vec, INT* len)=0;
		void get_int_vector(INT** vec, INT* len)=0;
		void get_shortreal_vector(SHORTREAL** vec, INT* len)=0;
		void get_real_vector(DREAL** vec, INT* len)=0;

		void get_byte_matrix(BYTE** matrix, INT* num_feat, INT* num_vec)=0;
		void get_int_matrix(INT** matrix, INT* num_feat, INT* num_vec)=0;
		void get_shortreal_matrix(SHORTREAL** matrix, INT* num_feat, INT* num_vec)=0;
		void get_real_matrix(DREAL** matrix, INT* num_feat, INT* num_vec)=0;

		void get_byte_sparsematrix(TSparse<BYTE>** matrix, INT* num_feat, INT* num_vec)=0;
		void get_int_sparsematrix(TSparse<INT>** matrix, INT* num_feat, INT* num_vec)=0;
		void get_shortreal_sparsematrix(TSparse<SHORTREAL>** matrix, INT* num_feat, INT* num_vec)=0;
		void get_real_sparsematrix(TSparse<DREAL>** matrix, INT* num_feat, INT* num_vec)=0;

		void get_string_list(T_STRING<CHAR>** strings, INT* num_str)=0;

		/** set functions - to pass data from shogun to the target interface */
		void create_return_values(INT num_val)=0;
		void set_byte_vector(BYTE* vec, INT len)=0;
		void set_int_vector(INT* vec, INT len)=0;
		void set_shortreal_vector(SHORTREAL* vec, INT len)=0;
		void set_real_vector(DREAL* vec, INT len)=0;

		void set_byte_matrix(BYTE* matrix, INT num_feat, INT num_vec)=0;
		void set_int_matrix(INT* matrix, INT num_feat, INT num_vec)=0;
		void set_shortreal_matrix(SHORTREAL* matrix, INT num_feat, INT num_vec)=0;
		void set_real_matrix(DREAL* matrix, INT num_feat, INT num_vec)=0;

		void set_byte_sparsematrix(TSparse<BYTE>* matrix, INT num_feat, INT num_vec)=0;
		void set_int_sparsematrix(TSparse<INT>* matrix, INT num_feat, INT num_vec)=0;
		void set_shortreal_sparsematrix(TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)=0;
		void set_real_sparsematrix(TSparse<DREAL>* matrix, INT num_feat, INT num_vec)=0;

		void set_string_list(T_STRING<CHAR>* strings, INT num_str)=0;

		void submit_return_values()=0;

	protected:
		INT arg_counter;
}:
