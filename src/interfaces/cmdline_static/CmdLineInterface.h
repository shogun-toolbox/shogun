#ifndef __CMDLINEINTERFACE__H_
#define __CMDLINEINTERFACE__H_

#include <shogun/base/DynArray.h>
#include <shogun/io/SGIO.h>

#include <shogun/ui/SGInterface.h>

namespace shogun
{
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
		virtual float64_t get_real();
		virtual bool get_bool();

		virtual char* get_string(int32_t& len);

		virtual void get_vector(uint8_t*& vec, int32_t& len);
		virtual void get_vector(char*& vec, int32_t& len);
		virtual void get_vector(int32_t*& vec, int32_t& len);
		virtual void get_vector(float32_t*& vec, int32_t& len);
		virtual void get_vector(float64_t*& vec, int32_t& len);
		virtual void get_vector(int16_t*& vec, int32_t& len);
		virtual void get_vector(uint16_t*& vec, int32_t& len);

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

		virtual bool create_return_values(int32_t num_val);
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

		/** @return object name */
		inline virtual const char* get_name() const { return "CmdLineInterface"; }

	private:
		DynArray<char*>* m_lhs;
		DynArray<char*>* m_rhs;
};
}
#endif // __CMDLINEINTERFACE__H_
