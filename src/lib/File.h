/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "lib/common.h"

/// A file consists of a header
/// then an alternation of a type header and data
///
/// or just raw data (simplefile=true)
///
/// the more complex stuff is currently not implemented
///
class CFile
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

	INT parse_first_header(EFeatureType &type);
	INT parse_next_header(EFeatureType &type);

	// set target to NULL to get it automagically allocated
	// set num to 0 if whole file is to be read
	INT*   load_int_data(INT* target, long& num);
	DREAL*  load_real_data(DREAL* target, long& num);
	CHAR*  load_char_data(CHAR* target, long& num);
	BYTE*  load_byte_data(BYTE* target, long& num);
	WORD*  load_word_data(WORD* target, long& num);
	SHORT* load_short_data(SHORT* target, long& num);

	bool save_int_data(INT* src, LONG num);
	bool save_real_data(DREAL* src, LONG num);
	bool save_char_data(CHAR* src, LONG num);
	bool save_byte_data(BYTE* src, LONG num);
	bool save_word_data(WORD* src, LONG num);
	bool save_short_data(SHORT* src, LONG num);

	inline bool is_ok()
	{
		return status;
	}

protected: 
	bool read_header();
	bool write_header();

protected: 
	FILE* file;
	bool status;
	CHAR task;
	CHAR* fname;
	EFeatureType expected_type;
	INT num_header;
	CHAR fourcc[4];
};
#endif
