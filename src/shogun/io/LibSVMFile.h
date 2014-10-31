/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Jiaolong Xu
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 * Written (W) 2010 Soeren Sonnenburg
 */

#ifndef __LIBSVMFILE_H__
#define __LIBSVMFILE_H__

#include <shogun/lib/config.h>
#include <shogun/io/File.h>

namespace shogun
{

class CDelimiterTokenizer;
class CLineReader;
class CParser;
template <class ST> class SGString;
template <class T> class SGSparseVector;

/** @brief read sparse real valued features in svm light format
 * e.g. -1 1:10.0 2:100.2 1000:1.3
 * with -1 == (optional) label
 * and dim 1    - value  10.0
 *     dim 2    - value 100.2
 *     dim 1000 - value   1.3
 */
class CLibSVMFile : public CFile
{
public:
	/** default constructor */
	CLibSVMFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CLibSVMFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CLibSVMFile(const char* fname, char rw='r', const char* name=NULL);

	/** destructor */
	virtual ~CLibSVMFile();

#ifndef SWIG // SWIG should skip this part
	
	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_sparse_matrix(
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_sparse_matrix(
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	void get_sparse_matrix(
			SGSparseVector<bool>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<int8_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<char>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix_feat, int32_t & num_feat, int32_t & num_vec,
			SGVector<float64_t>*& multilabel, int32_t & num_classes, bool load_labels=true);
	//@}
	
	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec,
			const SGVector<float64_t>* multilabel);
	//@}

#endif // #ifndef SWIG // SWIG should skip this part

	virtual const char* get_name() const { return "LibSVMFile"; }

private:
	/** class initialization */
	void init();

	/** class initialization */
	void init_with_defaults();

	/** get number of lines */
	int32_t get_num_lines();

	/** is it a feature entry */
	bool is_feat_entry(const SGVector<char> entry);
private:
	/** delimiter for index and data in sparse entries */
	char m_delimiter_feat;

	/** delimiter for multiple labels*/
	char m_delimiter_label;

	/** object for reading lines from file */
	CLineReader* m_line_reader;

	/** parser of lines */
	CParser* m_parser;

	/** tokenizer for line_reader */
	CDelimiterTokenizer* m_line_tokenizer;

	/** delimiter for parsing lines */
	CDelimiterTokenizer* m_whitespace_tokenizer;

	/** delimiter for parsing sparse entries */
	CDelimiterTokenizer* m_delimiter_feat_tokenizer;

	/** delimiter for parsing multiple labels */
	CDelimiterTokenizer* m_delimiter_label_tokenizer;
   };

}

#endif /** __LIBSVMFILE_H__ */
