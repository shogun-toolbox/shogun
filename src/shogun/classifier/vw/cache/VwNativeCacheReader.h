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

#ifndef _VW_NATIVECACHE_READ_H__
#define _VW_NATIVECACHE_READ_H__

#include <shogun/lib/config.h>
#include <shogun/classifier/vw/cache/VwCacheReader.h>

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
// Packed structure for efficient storage
struct one_float
{
	// The float to store
	float32_t f;
} __attribute__((packed));

#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Class CVwNativeCacheReader reads from a cache exactly as
 * that which has been produced by VW's default cache format.
 *
 * It is compatible with VW version 5.1.
 */
class CVwNativeCacheReader: public CVwCacheReader
{
public:
	/**
	 * Default constructor
	 */
	CVwNativeCacheReader();

	/**
	 * Constructor, opens a file whose name is specified
	 *
	 * @param fname file name
	 * @param env_to_use Environment to use
	 */
	CVwNativeCacheReader(char * fname, CVwEnvironment* env_to_use);

	/**
	 * Constructor, passed a file descriptor
	 *
	 * @param f descriptor of opened file
	 * @param env_to_use Environment to use
	 */
	CVwNativeCacheReader(int32_t f, CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwNativeCacheReader();

	/**
	 * Set the file descriptor to use
	 *
	 * @param f descriptor of cache file
	 */
	virtual void set_file(int32_t f);

	/**
	 * Read one cached example
	 *
	 * @return example as VwExample*
	 */
	virtual bool read_cached_example(VwExample* const ae);

	/**
	 * Check whether cache is readable
	 */
	void check_cache_metadata();

	/**
	 * Return the name of the object.
	 *
	 * @return VwNativeCacheReader
	 */
	virtual const char* get_name() const { return "VwNativeCacheReader"; }

private:
	/**
	 * Initialize members
	 */
	void init();

	/**
	 * Decode an int32_t from RLE-encoded data
	 *
	 * @param p pointer to data
	 * @param i decoded int
	 *
	 * @return new pointer position
	 */
	char* run_len_decode(char *p, vw_size_t& i);

	/**
	 * Decode a signed int32_t from an encoded unsigned form
	 *
	 * @param n encoded unsigned int
	 *
	 * @return decoded signed int
	 */
	inline int32_t ZigZagDecode(uint32_t n)
	{
		return (n >> 1) ^ -static_cast<int32_t>(n & 1);
	}

	/**
	 * Read a label from the buffer, helper
	 *
	 * @param ld label pointer
	 * @param c pointer to buffer
	 *
	 * @return new pointer position
	 */
	char* bufread_label(VwLabel* const ld, char* c);

	/**
	 * Read a label from the buffer
	 *
	 * @param ld label pointer
	 *
	 * @return number of bytes read
	 */
	vw_size_t read_cached_label(VwLabel* const ld);

	/**
	 * Read the tag of an example from the cache
	 *
	 * @param ae
	 *
	 * @return number of bytes read
	 */
	vw_size_t read_cached_tag(VwExample* const ae);


protected:
	/// Buffer to read from
	CIOBuffer buf;

private:
	// Used while parsing
	const vw_size_t char_size;
	vw_size_t neg_1;
	vw_size_t general;
};

}
#endif // _VW_NATIVECACHE_READ_H__
