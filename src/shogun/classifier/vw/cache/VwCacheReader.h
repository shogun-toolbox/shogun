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

#ifndef _VW_CACHEREAD_H__
#define _VW_CACHEREAD_H__

#include <shogun/lib/config.h>
#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>
#include <shogun/io/IOBuffer.h>
#include <shogun/classifier/vw/vw_common.h>

namespace shogun
{

/// Enum EVwCacheType specifies the type of
/// cache used, either C_NATIVE or C_PROTOBUF.
enum EVwCacheType
{
	C_NATIVE = 0,
	C_PROTOBUF = 1
};

/** @brief Base class from which all cache readers for VW
 * should be derived.
 *
 * The object is given cache file information and the
 * environment which will be used during parsing, and must
 * implement a read_cached_example() function which returns
 * a parsed example by reference.
 */
class CVwCacheReader: public CSGObject
{
public:
	/**
	 * Default constructor
	 */
	CVwCacheReader();

	/**
	 * Constructor, opens file specified by filename
	 *
	 * @param fname name of file to open
	 * @param env_to_use Environment to use
	 */
	CVwCacheReader(char * fname, CVwEnvironment* env_to_use);

	/**
	 * Constructor which takes an already opened file descriptor
	 * as argument.
	 *
	 * @param f file descriptor
	 * @param env_to_use VwEnvironment object to use
	 */
	CVwCacheReader(int32_t f, CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwCacheReader();

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
	 * Update min and max labels seen in the environment
	 *
	 * @param label current label based on which to update
	 */
	virtual void set_mm(float64_t label)
	{
		env->min_label = CMath::min(env->min_label, label);
		if (label != FLT_MAX)
			env->max_label = CMath::max(env->max_label, label);
	}

	/**
	 * A dummy function performing no operation in case training
	 * is not to be performed.
	 *
	 * @param label label
	 */
	virtual void noop_mm(float64_t label) { }

	/**
	 * Function which is actually called to update min and max labels
	 * Should be set to one of the functions implemented for this.
	 *
	 * @param label label based on which to update
	 */
	virtual void set_minmax(float64_t label)
	{
		set_mm(label);
	}

	/**
	 * Function to read one example from the cache
	 *
	 * @return read example
	 */
	virtual bool read_cached_example(VwExample* const ae) = 0;

	/**
	 * Return the name of the object
	 *
	 * @return VwCacheReader
	 */
	virtual const char* get_name() const { return "VwCacheReader"; }

protected:
	/// File descriptor
	int32_t fd;

	/// Environment
	CVwEnvironment* env;
};

}
#endif // _VW_CACHEREAD_H__
