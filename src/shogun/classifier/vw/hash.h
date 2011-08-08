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
 * Adaptation from Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_HASH_H__
#define _VW_HASH_H__

#include <stdint.h>
#include <sys/types.h>
#include <ctype.h>
#include <shogun/classifier/vw/vw_common.h>

namespace shogun
{
	const uint32_t hash_base = 97562527;

	/**
	 * Hash only strings, leaving integer values
	 * as they are
	 *
	 * @param s substring
	 * @param h seed value
	 *
	 * @return hashed value
	 */
	size_t hashstring (substring s, unsigned long h);

	/**
	 * Hash all with a standard algorithm
	 * even if it is numeric.
	 *
	 * @param s substring
	 * @param h seed value
	 *
	 * @return hashed value
	 */
	size_t hashall (substring s, unsigned long h);

	/**
	 * Returns the hash function, given the appropriate name as
	 * a string
	 *
	 * @param s string representing the hash function
	 *
	 * @return the hash function pointer
	 */
	hash_func_t getHasher(char* s);

	/**
	 * Uniform hash function
	 *
	 * @param key pointer to data
	 * @param length length
	 * @param initval seed
	 *
	 * @return hashed data as unsigned int
	 */
	uint32_t uniform_hash(const void *key, size_t length, uint32_t initval);
}

#endif // _VW_HASH_H__
