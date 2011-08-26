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
 * Written (W) 2011 Shashwat Lal Das
 * Adaptation of Vowpal Wabbit v5.1.
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society.
 */

#ifndef _VW_NATIVECACHE_WRITE_H__
#define _VW_NATIVECACHE_WRITE_H__

#include <shogun/classifier/vw/cache/VwCacheWriter.h>

namespace shogun
{
/** @brief Class CVwNativeCacheWriter writes a cache exactly as
 * that which would be produced by VW's default cache format.
 */
class CVwNativeCacheWriter: public CVwCacheWriter
{
public:
	/**
	 * Default constructor
	 */
	CVwNativeCacheWriter();

	/**
	 * Constructor, opens a file whose name is specified
	 *
	 * @param fname file name
	 * @param env_to_use Environment to use
	 */
	CVwNativeCacheWriter(char * fname, CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwNativeCacheWriter();

	/**
	 * Set the file descriptor to use
	 *
	 * @param f descriptor of cache file
	 */
	virtual void set_file(int32_t f);

	/**
	 * Cache one example
	 *
	 * @param ex example to write to cache
	 */
	virtual void cache_example(VwExample* &ex);

	/**
	 * Return the name of the object.
	 *
	 * @return VwNativeCacheWriter
	 */
	virtual const char* get_name() const { return "VwNativeCacheWriter"; }

private:
	/**
	 * Initialize members
	 */
	void init();

	/**
	 * Write the header of the cache.
	 * Includes version and weight bits information.
	 */
	void write_header();

	/**
	 * Use run-length encoding on an int
	 *
	 * @param p compressed data ptr
	 * @param i int32_t to compress
	 *
	 * @return ptr to compressed data
	 */
	char* run_len_encode(char *p, vw_size_t i);

	/**
	 * Encode a signed int32_t into an unsigned representation
	 *
	 * @param n signed int
	 *
	 * @return unsigned int
	 */
	inline uint32_t ZigZagEncode(int32_t n)
	{
		uint32_t ret = (n << 1) ^ (n >> 31);

		return ret;
	}

	/**
	 * Cache a label into the buffer, helper function
	 *
	 * @param ld label
	 * @param c pointer to last written buffer position
	 *
	 * @return new position of pointer
	 */
	char* bufcache_label(VwLabel* ld, char* c);

	/**
	 * Cache label into buffer
	 *
	 * @param ld label
	 */
	void cache_label(VwLabel* ld);

	/**
	 * Write the tag into the buffer
	 *
	 * @param tag tag
	 */
	void cache_tag(v_array<char> tag);

	/**
	 * Write a byte into the buffer
	 *
	 * @param s byte of data
	 */
	void output_byte(unsigned char s);

	/**
	 * Write the features into the buffer
	 *
	 * @param index namespace index
	 * @param begin first feature
	 * @param end pointer to end of features
	 */
	void output_features(unsigned char index, VwFeature* begin, VwFeature* end);

protected:
	/// IOBuffer used for writing
	CIOBuffer buf;

private:
	/// Used for encoding/decoding -1
	vw_size_t neg_1;
	/// Used for encoding/decoding other numbers
	vw_size_t general;
	/// int size for encoding/decoding
	vw_size_t int_size;
};

}
#endif // _VW_NATIVECACHE_WRITE_H__
