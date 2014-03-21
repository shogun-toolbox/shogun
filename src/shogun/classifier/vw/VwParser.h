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

#ifndef _VW_PARSER_H__
#define _VW_PARSER_H__

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Hash.h>
#include <shogun/classifier/vw/vw_common.h>
#include <shogun/classifier/vw/cache/VwCacheWriter.h>

namespace shogun
{
/// The type of input to parse
enum E_VW_PARSER_TYPE
{
	T_VW = 1,
	T_SVMLIGHT = 2,
	T_DENSE = 3
};

/** @brief CVwParser is the object which provides the
 * functions to parse examples from buffered input.
 *
 * An instance of this class can be created in
 * CStreamingVwFile and the appropriate read_*_features
 * function called to parse examples from different formats.
 *
 * It also encapsulates a CVwCacheWriter object which may
 * be used in case a cache file is to be generated simultaneously
 * with parsing.
 */
class CVwParser: public CSGObject
{
public:
	/**
	 * Default constructor
	 */
	CVwParser();

	/**
	 * Constructor taking environment as parameter.
	 *
	 * @param env_to_use CVwEnvironment to use
	 */
	CVwParser(CVwEnvironment* env_to_use);

	/**
	 * Destructor
	 */
	virtual ~CVwParser();

	/**
	 * Get the environment
	 *
	 * @return environment as CVwEnvironment*
	 */
	CVwEnvironment* get_env()
	{
		SG_REF(env);
		return env;
	}

	/**
	 * Set the environment
	 *
	 * @param env_to_use environment as CVwEnvironment*
	 */
	void set_env(CVwEnvironment* env_to_use)
	{
		env = env_to_use;
		SG_REF(env);
	}

	/**
	 * Set the cache parameters
	 *
	 * @param fname name of the cache file
	 * @param type type of cache as one in EVwCacheType
	 */
	void set_cache_parameters(char * fname, EVwCacheType type = C_NATIVE)
	{
		init_cache(fname, type);
	}

	/**
	 * Return the type of cache
	 *
	 * @return cache type as EVwCacheType
	 */
	EVwCacheType get_cache_type()
	{
		return cache_type;
	}

	/**
	 * Set whether to write cache file or not
	 *
	 * @param wr_cache write cache or not
	 */
	void set_write_cache(bool wr_cache)
	{
		write_cache = wr_cache;
		if (wr_cache)
			init_cache(NULL);
		else
			if (cache_writer)
				SG_UNREF(cache_writer);
	}

	/**
	 * Return whether cache will be written or not
	 *
	 * @return will cache be written?
	 */
	bool get_write_cache()
	{
		return write_cache;
	}

	/**
	 * Update min and max labels seen in the environment
	 *
	 * @param label current label based on which to update
	 */
	void set_mm(float64_t label)
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
	void noop_mm(float64_t label) { }

	/**
	 * Function which is actually called to update min and max labels
	 * Should be set to one of the functions implemented for this.
	 *
	 * @param label label based on which to update
	 */
	void set_minmax(float64_t label)
	{
		set_mm(label);
	}

	/**
	 * Reads input from the buffer and parses it into a VwExample
	 *
	 * @param buf IOBuffer which contains input
	 * @param ex parsed example
	 *
	 * @return number of characters read for this example
	 */
	int32_t read_features(CIOBuffer* buf, VwExample*& ex);

	/**
	 * Read an example from an SVMLight file
	 *
	 * @param buf IOBuffer which contains input
	 * @param ae parsed example
	 *
	 * @return number of characters read for this example
	 */
	int32_t read_svmlight_features(CIOBuffer* buf, VwExample*& ae);

	/**
	 * Read an example from a file with dense vectors
	 *
	 * @param buf IOBuffer which contains input
	 * @param ae parsed example
	 *
	 * @return number of characters read for this example
	 */
	int32_t read_dense_features(CIOBuffer* buf, VwExample*& ae);

	/**
	 * Return the name of the object
	 *
	 * @return VwParser
	 */
	virtual const char* get_name() const { return "VwParser"; }

protected:
	/**
	 * Initialize the cache writer
	 *
	 * @param fname cache file name
	 * @param type cache type as EVwCacheType, default is C_NATIVE
	 */
	void init_cache(char * fname, EVwCacheType type = C_NATIVE);

	/**
	 * Get value of feature from a given substring.
	 * A default of 1 is assumed if no explicit value is specified.
	 *
	 * @param s substring, usually a feature:value string
	 * @param name returned array of substrings, split into name and value
	 * @param v value of feature, set by reference
	 */
	void feature_value(substring &s, v_array<substring>& name, float32_t &v);

	/**
	 * Split a given substring into an array of substrings
	 * based on a specified delimiter
	 *
	 * @param delim delimiter to use
	 * @param s substring to tokenize
	 * @param ret array of substrings, returned
	 */
	void tokenize(char delim, substring s, v_array<substring> &ret);

	/**
	 * Get the index of a character in a memory location
	 * taking care not to go beyond the max pointer.
	 *
	 * @param start start memory location, char*
	 * @param v character to search for
	 * @param max last location to look in
	 *
	 * @return index of found location as char*
	 */
	inline char* safe_index(char *start, char v, char *max)
	{
		while (start != max && *start != v)
			start++;
		return start;
	}

public:
	/// Hash function to use, of type hash_func_t
	hash_func_t hasher;

protected:
	/// Environment of VW - used by parser
	CVwEnvironment* env;
	/// Object which will be used for writing cache
	CVwCacheWriter* cache_writer;
	/// Type of cache
	EVwCacheType cache_type;
	/// Whether to write cache or not
	bool write_cache;

private:
	/// Used during parsing
	v_array<substring> channels;
	v_array<substring> words;
	v_array<substring> name;
};

}
#endif // _VW_PARSER_H__
