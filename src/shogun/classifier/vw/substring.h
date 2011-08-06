/*
 * Copyright (c) 2009 Yahoo! Inc.  All rights reserved.  The copyrights
 * embodied in the content of this file are licensed under the BSD
 * (revised) open source license.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */
#ifndef _VW_SUBSTRING_H__
#define _VW_SUBSTRING_H__

#include <shogun/lib/DataType.h>
#include <shogun/io/SGIO.h>

#include <stdlib.h>
#include <string>

namespace shogun
{
/**
 * @brief struct Substring, specified by
 * start position and end position.
 *
 * Used to mark strings in a buffer, where they
 * need not be delimited by NUL characters.
 */
struct substring
{
	char *start;
	char *end;
};

/**
 * Return a C string from the substring
 * @param s substring
 * @return new C string representation
 */
inline char* c_string_of_substring(substring s)
{
	index_t len = s.end - s.start+1;
	char* ret = SG_CALLOC(char, len);
	memcpy(ret,s.start,len-1);
	return ret;
}

/**
 * Print the substring
 * @param s substring
 */
inline void print_substring(substring s)
{
	SG_SPRINT("%s\n", s.start,s.end - s.start);
}

/**
 * Get value of substring as float
 * (if possible)
 * @param s substring
 * @return float32_t value of substring
 */
inline float32_t float_of_substring(substring s)
{
	char* endptr = s.end;
	float32_t f = strtof(s.start,&endptr);
	if (endptr == s.start && s.start != s.end)
		SG_SERROR("error: %s is not a float!\n", std::string(s.start, s.end-s.start).c_str());

	return f;
}

/**
 * Return value of substring as double
 * @param s substring
 * @return substring as double
 */
inline float32_t double_of_substring(substring s)
{
	char* endptr = s.end;
	float32_t f = strtod(s.start,&endptr);
	if (endptr == s.start && s.start != s.end)
		SG_SERROR("Error!:%s is not a double!\n", std::string(s.start, s.end-s.start).c_str());

	return f;
}

/**
 * Integer value of substring
 * @param s substring
 * @return int value of substring
 */
inline int int_of_substring(substring s)
{
	return atoi(std::string(s.start, s.end-s.start).c_str());
}

/**
 * Unsigned long value of substring
 * @param s substring
 * @return unsigned long value of substring
 */
inline unsigned long ulong_of_substring(substring s)
{
	return strtoul(std::string(s.start, s.end-s.start).c_str(),NULL,10);
}

/**
 * Length of substring
 * @param s substring
 * @return length of substring
 */
inline unsigned long ss_length(substring s)
{
	return (s.end - s.start);
}
}
#endif // _VW_SUBSTRING_H__
