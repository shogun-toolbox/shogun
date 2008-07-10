/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lib/common.h"
#include "lib/DynamicArray.h"
#include "base/SGObject.h"

template <class ST> struct T_STRING;
template <class ST> struct TSparse;

/** A File access class. A file consists of a fourcc header then an alternation
 * of a type header and data or just raw data (simplefile=true). However this
 * implementation is not complete - the more complex stuff is currently not
 * implemented.
 */

class CFile : public CSGObject
{
public:
	/** constructor
	 *
	 * @param f already opened file
	 */
	CFile(FILE* f);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param type specifies the datatype used in the file (F_INT,...)
	 * @param fourcc in the case fourcc is 0, type will be ignored
	 *               and the file is treated as if it has a
	 *               header/[typeheader,data]+ else the files header
	 *               will be checked to contain the specified fourcc
	 *               (e.g. 'RFEA')
	 */
	CFile(CHAR* fname, CHAR rw, EFeatureType type, CHAR fourcc[4]=NULL);

	~CFile();

	/** parse first header - defunct!
	 *
	 * @param type feature type
	 * @return -1
	 */
	INT parse_first_header(EFeatureType &type);

	/** parse next header - defunct!
	 *
	 * @param type feature type
	 * @return -1
	 */
	INT parse_next_header(EFeatureType &type);

	// set target to NULL to get it automagically allocated
	// set num to 0 if whole file is to be read
	/** load integer data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	INT*   load_int_data(INT* target, LONG& num);

	/** load real data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	DREAL*  load_real_data(DREAL* target, LONG& num);

	/** load shortreal data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	SHORTREAL*  load_shortreal_data(SHORTREAL* target, LONG& num);

	/** load char data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	CHAR*  load_char_data(CHAR* target, LONG& num);

	/** load byte data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	BYTE*  load_byte_data(BYTE* target, LONG& num);

	/** load word data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	WORD*  load_word_data(WORD* target, LONG& num);

	/** load short data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	SHORT* load_short_data(SHORT* target, LONG& num);

	/** save integer data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_int_data(INT* src, LONG num);

	/** save real data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_real_data(DREAL* src, LONG num);

	/** save shortreal data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_shortreal_data(SHORTREAL* src, LONG num);

	/** save char data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_char_data(CHAR* src, LONG num);

	/** save byte data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_byte_data(BYTE* src, LONG num);

	/** save word data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_word_data(WORD* src, LONG num);

	/** save short data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_short_data(SHORT* src, LONG num);

	/** check if status is ok
	 *
	 * @return whether status is ok
	 */
	inline bool is_ok()
	{
		return status;
	}


	/** read sparse real valued features in svm light format
	 * e.g. -1 1:10.0 2:100.2 1000:1.3 
	 * with -1 == (optional) label
	 * and dim 1    - value  10.0
	 *     dim 2    - value 100.2
	 *     dim 1000 - value   1.3
	 *
	 * @param matrix matrix to read into
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectors in matrix
	 * @return if reading was successful
	 */
	bool read_real_valued_sparse(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec);

	/** write sparse real valued features in svm light format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_sparse(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec);

	/** read dense real valued features, simple ascii format
	 * e.g. 1.0 1.1 0.2 
	 *      2.3 3.5 5
	 *
	 *  a matrix that consists of 3 vectors with each of 2d
	 *
	 * @param matrix matrix to read into
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectors in matrix
	 * @return if reading was successful
	 */
	bool read_real_valued_dense(DREAL*& matrix, INT& num_feat, INT& num_vec);

	/** write dense real valued features, simple ascii format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_dense(const DREAL* matrix, INT num_feat, INT num_vec);

	/** read char string features, simple ascii format
	 * e.g. foo bar
	 *      ACGTACGTATCT
	 *
	 *  two strings
	 *
	 * @param strings strings to read into
	 * @param num_str number of strings
	 * @param max_string_len length of longest string
	 * @return if reading was successful
	 */
	bool read_char_valued_strings(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len);

	/** write char string features, simple ascii format
	 *
	 * @param strings strings to write
	 * @param num_str number of strings
	 * @return if writing was successful
	 */
	bool write_char_valued_strings(const T_STRING<CHAR>* strings, INT num_str);

protected:
	/** read header
	 *
	 * @return whether operation was successful
	 */
	bool read_header();
	/** write header
	 *
	 * @return whether operation was successful
	 */
	bool write_header();

private:
	/** helper function to read_*valued_* */
	template <class T> void append_item(CDynamicArray<T>* items, CHAR* ptr_data, CHAR* ptr_item);

protected:
	/** file object */
	FILE* file;
	/** status */
	bool status;
	/** task */
	CHAR task;
	/** name of the handled file */
	CHAR* filename;
	/** expected feature type */
	EFeatureType expected_type;
	/** number of headers */
	INT num_header;
	/** fourcc */
	CHAR fourcc[4];
};
#endif
