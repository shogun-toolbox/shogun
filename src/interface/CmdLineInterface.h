#ifndef __CMDLINEINTERFACE__H_
#define __CMDLINEINTERFACE__H_

#include "lib/config.h"

#if defined(HAVE_CMDLINE)
#include "interface/SGInterface.h"

#define CMDLINE_COMMENT0 '#'
#define CMDLINE_COMMENT1 '%'

class CCmdLineInterface : public CSGInterface
{
	public:
		CCmdLineInterface();
		~CCmdLineInterface();

		/// reset to clean state
		virtual void reset(const char* line=NULL);

		/** get functions - to pass data from the target interface to shogun */

		/// get type of current argument (does not increment argument counter)
		virtual IFType get_argument_type();

		virtual int32_t get_int();
		virtual DREAL get_real();
		virtual bool get_bool();

		virtual char* get_string(int32_t& len);

		virtual void get_byte_vector(uint8_t*& vec, int32_t& len);
		virtual void get_char_vector(char*& vec, int32_t& len);
		virtual void get_int_vector(int32_t*& vec, int32_t& len);
		virtual void get_shortreal_vector(SHORTREAL*& vec, int32_t& len);
		virtual void get_real_vector(DREAL*& vec, int32_t& len);
		virtual void get_short_vector(SHORT*& vec, int32_t& len);
		virtual void get_word_vector(uint16_t*& vec, int32_t& len);

		virtual void get_byte_matrix(uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_char_matrix(char*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_int_matrix(int32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_shortreal_matrix(SHORTREAL*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_real_matrix(DREAL*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_short_matrix(SHORT*& matrix, int32_t& num_feat, int32_t& num_vec);
		virtual void get_word_matrix(uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_shortreal_ndarray(SHORTREAL*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_real_ndarray(DREAL*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_short_ndarray(SHORT*& array, int32_t*& dims, int32_t& num_dims);
		virtual void get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims);

		virtual void get_real_sparsematrix(TSparse<DREAL>*& matrix, int32_t& num_feat, int32_t& num_vec);

		virtual void get_byte_string_list(T_STRING<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len);
		virtual void get_char_string_list(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len);
		virtual void get_int_string_list(T_STRING<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len);
		virtual void get_short_string_list(T_STRING<SHORT>*& strings, int32_t& num_str, int32_t& max_string_len);
		virtual void get_word_string_list(T_STRING<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len);

		/** set functions - to pass data from shogun to the target interface */
		virtual void set_int(int32_t scalar);
		virtual void set_real(DREAL scalar);
		virtual void set_bool(bool scalar);

		virtual bool create_return_values(int32_t num_val);
		virtual void set_byte_vector(const uint8_t* vec, int32_t len);
		virtual void set_char_vector(const char* vec, int32_t len);
		virtual void set_int_vector(const int32_t* vec, int32_t len);
		virtual void set_shortreal_vector(const SHORTREAL* vec, int32_t len);
		virtual void set_real_vector(const DREAL* vec, int32_t len);
		virtual void set_short_vector(const SHORT* vec, int32_t len);
		virtual void set_word_vector(const uint16_t* vec, int32_t len);

		virtual void set_byte_matrix(const uint8_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_char_matrix(const char* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_int_matrix(const int32_t* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_shortreal_matrix(const SHORTREAL* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_real_matrix(const DREAL* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_short_matrix(const SHORT* matrix, int32_t num_feat, int32_t num_vec);
		virtual void set_word_matrix(const uint16_t* matrix, int32_t num_feat, int32_t num_vec);

		virtual void set_real_sparsematrix(const TSparse<DREAL>* matrix, int32_t num_feat, int32_t num_vec, LONG nnz);

		virtual void set_byte_string_list(const T_STRING<uint8_t>* strings, int32_t num_str);
		virtual void set_char_string_list(const T_STRING<char>* strings, int32_t num_str);
		virtual void set_int_string_list(const T_STRING<int32_t>* strings, int32_t num_str);
		virtual void set_short_string_list(const T_STRING<SHORT>* strings, int32_t num_str);
		virtual void set_word_string_list(const T_STRING<uint16_t>* strings, int32_t num_str);

		void* get_return_values();

		/** determine if given line is a comment or empty */
		bool skip_line(const char* line=NULL);

		/// get line from file or stdin or...
		char* get_line(FILE* infile=stdin, bool interactive_mode=true);

		/// parse a single line
		bool parse_line(char* line);

		/// print interactive prompt
		void print_prompt();

	private:
		const char* get_arg_increment()
		{
			ASSERT(m_rhs_counter>=0 && m_rhs_counter<m_nrhs+1); // +1 for action
			char* element=m_rhs->get_element(m_rhs_counter);
			m_rhs_counter++;

			return element;
		}

		const char* set_arg_increment()
		{
			ASSERT(m_lhs_counter>=0 && m_lhs_counter<m_nlhs+1); // +1 for action
			char* element=m_lhs->get_element(m_lhs_counter);
			m_lhs_counter++;

			return element;
		}

	private:
		CDynamicArray<char*>* m_lhs;
		CDynamicArray<char*>* m_rhs;
};
#endif // HAVE_CMDLINE
#endif // __CMDLINEINTERFACE__H_
