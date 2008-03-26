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
#include "base/SGObject.h"

/// A file consists of a header
/// then an alternation of a type header and data
///
/// or just raw data (simplefile=true)
///
/// the more complex stuff is currently not implemented
///
class CFile : public CSGObject
{
public:
	///Open a file of name fname with mode rw (r or w)
	/// - type specifies the datatype used in the file (F_INT,...)
	/// - fourcc : in the case fourcc is 0, type will be ignored
	/// and the file is treated as if it has a header/[typeheader,data]+
	/// else a the files header will be checked to contain the specified
	/// fourcc (e.g. 'RFEA')
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
