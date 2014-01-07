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

#ifndef _VW_CACHEWRITE_H__
#define _VW_CACHEWRITE_H__

#include <base/SGObject.h>
#include <lib/common.h>
#include <io/IOBuffer.h>
#include <classifier/vw/vw_common.h>
#include <classifier/vw/cache/VwCacheReader.h>

namespace shogun
{

/** @brief CVwCacheWriter is the base class for all VW cache creating
 * classes.
 *
 * The derived class must implement a cache_example() function which
 * writes that example into the cache file.
 * The class is provided with the file and the environment.
 */
class CVwCacheWriter: public CSGObject
{
public:

	/**
	 * Default constructor
	 */
	CVwCacheWriter();

	/**
	 * Constructor, opens file specified by filename
	 *
	 * @param fname name of file to open
	 * @param env_to_use environment
	 */
	CVwCacheWriter(char * fname, CVwEnvironment* env_to_use);

	/**
	 * Constructor, uses file specified by descriptor
	 *
	 * @param f descriptor of opened cache file
	 * @param env_to_use environment
	 */
	CVwCacheWriter(int32_t f, CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwCacheWriter();

	/**
	 * Set the file descriptor to use
	 *
	 * @param f descriptor of cache file
	 */
	virtual void set_file(int32_t f);

	/**
	 * Set the environment
	 *
	 * @param env_to_use environment
	 */
	virtual void set_env(CVwEnvironment* env_to_use);

	/**
	 * Get the environment
	 *
	 * @return environment
	 */
	virtual CVwEnvironment* get_env();

	/**
	 * Function to cache one example to the file
	 *
	 * @param ex example to cache
	 */
	virtual void cache_example(VwExample* &ex) = 0;

protected:

	/// File descriptor
	int32_t fd;

	/// Environment
	CVwEnvironment* env;
};

}
#endif
