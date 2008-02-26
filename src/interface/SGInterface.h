#ifndef __SGINTERFACE__H_
#define __SGINTERFACE__H_

#include "lib/common.h"
#include "base/SGObject.h"
#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"

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

class CSGInterface : public CSGObject
{
	public:
		CSGInterface();
		~CSGInterface();

		/// get action name, like 'send_command', 'get_svm' etc
		inline CHAR* get_action(INT &len)
		{
			ASSERT(arg_counter==0);
			if (m_nrhs<=0)
				SG_SERROR("No input arguments supplied.");

			return get_string(len);
		}

		/** get functions - to pass data from the target interface to shogun */
		virtual void parse_args(INT num_args, INT num_default_args)=0;

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type()=0;

		virtual INT get_int()=0;
		virtual DREAL get_real()=0;
		virtual bool get_bool()=0;

		virtual CHAR* get_string(INT& len)=0;
		virtual INT get_int_from_string();
		virtual DREAL get_real_from_string();
		virtual bool get_bool_from_string();

		//get_byte_vector(BYTE*& bytes, INT& len);
		virtual void get_byte_vector(BYTE** vec, INT* len)=0;
		virtual void get_int_vector(INT** vec, INT* len)=0;
		virtual void get_shortreal_vector(SHORTREAL** vec, INT* len)=0;
		virtual void get_real_vector(DREAL** vec, INT* len)=0;

		virtual void get_byte_matrix(BYTE** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_int_matrix(INT** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_shortreal_matrix(SHORTREAL** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_real_matrix(DREAL** matrix, INT* num_feat, INT* num_vec)=0;

		virtual void get_byte_sparsematrix(TSparse<BYTE>** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_int_sparsematrix(TSparse<INT>** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_shortreal_sparsematrix(TSparse<SHORTREAL>** matrix, INT* num_feat, INT* num_vec)=0;
		virtual void get_real_sparsematrix(TSparse<DREAL>** matrix, INT* num_feat, INT* num_vec)=0;

		virtual void get_string_list(T_STRING<CHAR>** strings, INT* num_str)=0;

		/** set functions - to pass data from shogun to the target interface */
		virtual void create_return_values(INT num_val)=0;
		virtual void set_byte_vector(BYTE* vec, INT len)=0;
		virtual void set_int_vector(INT* vec, INT len)=0;
		virtual void set_shortreal_vector(SHORTREAL* vec, INT len)=0;
		virtual void set_real_vector(DREAL* vec, INT len)=0;

		virtual void set_byte_matrix(BYTE* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_matrix(INT* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_matrix(SHORTREAL* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_real_matrix(DREAL* matrix, INT num_feat, INT num_vec)=0;

		virtual void set_byte_sparsematrix(TSparse<BYTE>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_sparsematrix(TSparse<INT>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_sparsematrix(TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_real_sparsematrix(TSparse<DREAL>* matrix, INT num_feat, INT num_vec)=0;

		virtual void set_string_list(T_STRING<CHAR>* strings, INT num_str)=0;

		virtual void submit_return_values()=0;

		/// general interface handler
		bool handle();

	protected:
		/// return true if str starts with cmd
		/// cmd is a 0 terminated string const
		/// str is a string of length len (not 0 terminated)
		static bool strmatch(CHAR* str, UINT len, const CHAR* cmd)
		{
			return (len>=strlen(cmd) 
					&& !strncmp(str, cmd, strlen(cmd)));
		}

	protected:
		INT arg_counter;
		INT m_nlhs;
		INT m_nrhs;
};

#endif // __SGINTERFACE__H_
