#ifndef __SGINTERFACE__H_
#define __SGINTERFACE__H_

#include "lib/config.h"

#if !defined(HAVE_SWIG)

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

		/* actions */
		/** calculate CRC sum */
		bool a_crc();
		/** translater string */
		bool a_translate_string();
		/** best path 2struct */
		bool a_best_path_2struct();
		/** best path trans */
		bool a_best_path_trans();
		/** best path trans deriv */
		bool a_best_path_trans_deriv();
		/** best path no b */
		bool a_best_path_no_b();
		/** best path trans simple */
		bool a_best_path_trans_simple();
		/** best path no b trans */
		bool a_best_path_no_b_trans();
		/** get version */
		bool a_get_version();
		/** set labels */
		bool a_set_labels();
		/** get labels */
		bool a_get_labels();
		/** obtain from position list */
		bool a_obtain_from_position_list();
		/** get features */
		bool a_get_features();
		/** add features */
		bool a_add_features();
		/** set features */
		bool a_set_features();
		/** get distance matrix */
		bool a_get_distance_matrix();
		/** get kernel matrix */
		bool a_get_kernel_matrix();
		/** set custom kernel */
		bool a_set_custom_kernel();
		/** set WD position weights */
		bool a_set_WD_position_weights();
		/** set subkernel weights */
		bool a_set_subkernel_weights();
		/** set subkernel weights combined */
		bool a_set_subkernel_weights_combined();
		/** set last subkernel weights */
		bool a_set_last_subkernel_weights();
		/** get SPEC consensus */
		bool a_get_SPEC_consensus();
		/** get SPEC scoring */
		bool a_get_SPEC_scoring();
		/** get WD consensus */
		bool a_get_WD_consensus();
		/** compute POIM WD */
		bool a_compute_POIM_WD();
		/** get WD scoring */
		bool a_get_WD_scoring();
		/** get WD position weights */
		bool a_get_WD_position_weights();
		/** get last subkernel weights */
		bool a_get_last_subkernel_weights();
		/** compute by subkernels */
		bool a_compute_by_subkernels();
		/** get kernel optimization */
		bool a_get_kernel_optimization();
		/** plugin estimate classify one example */
		bool a_plugin_estimate_classify_example();
		/** plugin estimate classify */
		bool a_plugin_estimate_classify();
		/** set plugin estimate */
		bool a_set_plugin_estimate();
		/** get plugin estimate */
		bool a_get_plugin_estimate();
		/** classify */
		bool a_classify();
		/** classify example */
		bool a_classify_example();
		/** get classifier */
		bool a_get_classifier();
		/** get SVM */
		bool a_get_svm();
		/** set SVM */
		bool a_set_svm();
		/** get SVM objective */
		bool a_get_svm_objective();
		/** compute HMM relative entropy */
		bool a_relative_entropy();
		/** compute HMM entropy */
		bool a_entropy();
		/** HMM classify */
		bool a_hmm_classify();
		/** HMM classify for a single example */
		bool a_hmm_classify_example();
		/** LinearHMM classify for 1-class examples */
		bool a_one_class_linear_hmm_classify();
		/** HMM classify for 1-class examples */
		bool a_one_class_hmm_classify();
		/** HMM classify for a single 1-class example */
		bool a_one_class_hmm_classify_example();
		/** get HMM's likelihood */
		bool a_hmm_likelihood();
		/** get HMM's Viterbi Path */
		bool a_get_viterbi_path();
		/** get HMM */
		bool a_get_hmm();
		/** append HMM */
		bool a_append_hmm();
		/** set HMM */
		bool a_set_hmm();
		/** issue help message */
		bool a_help();
		/** test function */
		bool a_test();

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

		virtual void get_byte_vector(BYTE*& vector, INT& len)=0;
		virtual void get_char_vector(CHAR*& vector, INT& len)=0;
		virtual void get_int_vector(INT*& vector, INT& len)=0;
		virtual void get_real_vector(DREAL*& vector, INT& len)=0;
		virtual void get_shortreal_vector(SHORTREAL*& vector, INT& len)=0;
		virtual void get_short_vector(SHORT*& vector, INT& len)=0;
		virtual void get_word_vector(WORD*& vector, INT& len)=0;


		virtual void get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)=0;


		virtual void get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec)=0;
		virtual void get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec)=0;


		virtual void get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len)=0;
		virtual void get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len)=0;


		/** set functions - to pass data from shogun to the target interface */
		virtual void create_return_values(INT num_val)=0;

		virtual void set_byte_vector(const BYTE* vec, INT len)=0;
		virtual void set_char_vector(const CHAR* vec, INT len)=0;
		virtual void set_int_vector(const INT* vec, INT len)=0;
		virtual void set_shortreal_vector(const SHORTREAL* vec, INT len)=0;
		virtual void set_real_vector(const DREAL* vec, INT len)=0;
		virtual void set_short_vector(const SHORT* vec, INT len)=0;
		virtual void set_word_vector(const WORD* vec, INT len)=0;


		virtual void set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)=0;


		virtual void set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec)=0;
		virtual void set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec)=0;


		virtual void set_string_list(const T_STRING<CHAR>* strings, INT num_str)=0;
		virtual void set_string_list(const T_STRING<WORD>* strings, INT num_str)=0;


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

		/// get action name, like 'send_command', 'get_svm' etc
		inline CHAR* get_action(INT &len)
		{
			ASSERT(m_rhs_counter==0);
			if (m_nrhs<=0)
				SG_SERROR("No input arguments supplied.");

			return get_string(len);
		}

	protected:
		INT m_lhs_counter;
		INT m_rhs_counter;
		INT m_nlhs;
		INT m_nrhs;

	private:
		/** helper function for hmm classify */
		bool do_hmm_classify(bool linear=false, bool one_class=false);
		/** helper function for hmm classify on 1 example */
		bool do_hmm_classify_example(bool one_class=false);
		/** helper function for add/set features */
		bool do_set_features(bool add=false);
};

typedef bool (CSGInterface::*CSGInterfacePtr)();

typedef struct {
	CHAR* action;
	CSGInterfacePtr method;
	CHAR* usage;
} CSGInterfaceMethod;

#endif // !HAVE_SWIG
#endif // __SGINTERFACE__H_
