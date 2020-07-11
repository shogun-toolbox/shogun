/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Yuyu Zhang, Viktor Gal, 
 *          Thoralf Klein, Fernando Iglesias, Sergey Lisitsyn
 */
#ifndef __STREAMING_ASCIIFILE_H__
#define __STREAMING_ASCIIFILE_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFile.h>
#include <shogun/lib/v_array.h>

namespace shogun
{

struct substring;
template <class ST> struct SGSparseVectorEntry;

/** @brief Class StreamingAsciiFile to read vector-by-vector from ASCII files.
 *
 * The object must be initialized like a CSVFile.
 */
class StreamingAsciiFile: public StreamingFile
{

public:
	/**
	 * Default constructor
	 *
	 */
	StreamingAsciiFile();

	/**
	 * Constructor taking file name argument
	 *
	 * @param fname file name
	 * @param rw read/write mode
	 */
	StreamingAsciiFile(const char* fname, char rw='r');

	/**
	 * Destructor
	 */
	~StreamingAsciiFile() override;

	/** set delimiting character
	 *
	 * @param delimiter the character used as delimiter
	 */
	void set_delimiter(char delimiter);

#ifndef SWIG // SWIG should skip this
	/**
	 * Utility function to convert a string to a boolean value
	 *
	 * @param str string to convert
	 *
	 * @return boolean value
	 */
	inline bool str_to_bool(char *str)
	{
		return (atoi(str)!=0);
	}

#define GET_VECTOR_DECL(sg_type)					\
	void get_vector						\
		(sg_type*& vector, int32_t& len) override;			\
									\
	void get_vector_and_label				\
		(sg_type*& vector, int32_t& len, float64_t& label) override;	\
									\
	void get_string						\
		(sg_type*& vector, int32_t& len) override;			\
									\
	void get_string_and_label			\
		(sg_type*& vector, int32_t& len, float64_t& label) override;	\
									\
	void get_sparse_vector					\
		(SGSparseVectorEntry<sg_type>*& vector, int32_t& len) override;	\
									\
	void get_sparse_vector_and_label			\
		(SGSparseVectorEntry<sg_type>*& vector, int32_t& len, float64_t& label) override;

	GET_VECTOR_DECL(bool)
	GET_VECTOR_DECL(uint8_t)
	GET_VECTOR_DECL(char)
	GET_VECTOR_DECL(int32_t)
	GET_VECTOR_DECL(float32_t)
	GET_VECTOR_DECL(float64_t)
	GET_VECTOR_DECL(int16_t)
	GET_VECTOR_DECL(uint16_t)
	GET_VECTOR_DECL(int8_t)
	GET_VECTOR_DECL(uint32_t)
	GET_VECTOR_DECL(int64_t)
	GET_VECTOR_DECL(uint64_t)
	GET_VECTOR_DECL(floatmax_t)
#undef GET_VECTOR_DECL

#endif // #ifndef SWIG // SWIG should skip this

	/** @return object name */
	const char* get_name() const override
	{
		return "StreamingAsciiFile";

	}

private:
	/** helper function to read vectors / matrices
	 *
	 * @param items dynamic array of values
	 * @param ptr_data
	 * @param ptr_item
	 */
	template <class T> void append_item(std::vector<T>& items, char* ptr_data, char* ptr_item);

	/**
	 * Split a given substring into an array of substrings
	 * based on a specified delimiter
	 *
	 * @param delim delimiter to use
	 * @param s substring to tokenize
	 * @param ret array of substrings, returned
	 */
	void tokenize(char delim, substring s, v_array<substring> &ret);

private:
	/// Helper for parsing
	v_array<substring> words;

	/** delimiter */
	char m_delimiter;
};
}
#endif //__STREAMING_ASCIIFILE_H__
