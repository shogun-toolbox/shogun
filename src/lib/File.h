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

#include "base/SGObject.h"
#include "lib/DynamicArray.h"
#include "features/Features.h"

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
	CFile(char* fname, char rw, EFeatureType type, char fourcc[4]=NULL);

	~CFile();

	/** parse first header - defunct!
	 *
	 * @param type feature type
	 * @return -1
	 */
	int32_t parse_first_header(EFeatureType &type);

	/** parse next header - defunct!
	 *
	 * @param type feature type
	 * @return -1
	 */
	int32_t parse_next_header(EFeatureType &type);

	// set target to NULL to get it automagically allocated
	// set num to 0 if whole file is to be read
	/** load integer data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	int32_t*   load_int_data(int32_t* target, int64_t& num);

	/** load real data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	float64_t*  load_real_data(float64_t* target, int64_t& num);

	/** load shortreal data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	float32_t*  load_shortreal_data(float32_t* target, int64_t& num);

	/** load char data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	char*  load_char_data(char* target, int64_t& num);

	/** load byte data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	uint8_t*  load_byte_data(uint8_t* target, int64_t& num);

	/** load word data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	uint16_t*  load_word_data(uint16_t* target, int64_t& num);

	/** load short data
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	int16_t* load_short_data(int16_t* target, int64_t& num);

	/** save integer data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_int_data(int32_t* src, int64_t num);

	/** save real data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_real_data(float64_t* src, int64_t num);

	/** save shortreal data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_shortreal_data(float32_t* src, int64_t num);

	/** save char data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_char_data(char* src, int64_t num);

	/** save byte data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_byte_data(uint8_t* src, int64_t num);

	/** save word data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_word_data(uint16_t* src, int64_t num);

	/** save short data
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	bool save_short_data(int16_t* src, int64_t num);

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
	bool read_real_valued_sparse(
		TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);

	/** write sparse real valued features in svm light format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_sparse(
		const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec);

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
	bool read_real_valued_dense(
		float64_t*& matrix, int32_t& num_feat, int32_t& num_vec);

	/** write dense real valued features, simple ascii format
	 *
	 * @param matrix matrix to write
	 * @param num_feat number of features for each vector
	 * @param num_vec number of vectros in matrix
	 * @return if writing was successful
	 */
	bool write_real_valued_dense(
		const float64_t* matrix, int32_t num_feat, int32_t num_vec);

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
	bool read_char_valued_strings(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len);

	/** write char string features, simple ascii format
	 *
	 * @param strings strings to write
	 * @param num_str number of strings
	 * @return if writing was successful
	 */
	bool write_char_valued_strings(const T_STRING<char>* strings, int32_t num_str);

	/** @return object name */
	inline virtual const char* get_name() { return "File"; }

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
	template <class T> void append_item(CDynamicArray<T>* items, char* ptr_data, char* ptr_item);

protected:
	/** file object */
	FILE* file;
	/** status */
	bool status;
	/** task */
	char task;
	/** name of the handled file */
	char* filename;
	/** expected feature type */
	EFeatureType expected_type;
	/** number of headers */
	int32_t num_header;
	/** fourcc */
	char fourcc[4];
};
#endif
