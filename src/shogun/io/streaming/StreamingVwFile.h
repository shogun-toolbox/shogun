/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_VWFILE_H__
#define __STREAMING_VWFILE_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFile.h>
#include <shogun/classifier/vw/vw_common.h>
#include <shogun/classifier/vw/VwParser.h>

namespace shogun
{
/// Parse function typedef. Takes an IOBuffer and VwExample as arguments.
typedef int32_t (CVwParser::*parse_func)(CIOBuffer*, VwExample*&);

/** @brief Class StreamingVwFile to read vector-by-vector from
 * Vowpal Wabbit data files.
 * It reads the example and label into one object of VwExample type.
*/
class CStreamingVwFile: public CStreamingFile
{
public:
	/**
	 * Default constructor
	 *
	 */
	CStreamingVwFile();

	/**
	 * Constructor taking file name argument
	 *
	 * @param fname file name
	 * @param rw read/write mode
	 */
	CStreamingVwFile(const char* fname, char rw='r');

	/**
	 * Destructor
	 */
	virtual ~CStreamingVwFile();

	/**
	 * Set the type of parser, i.e.,
	 * T_VW, T_SVMLIGHT or T_DENSE.
	 *
	 * @param type parser type as enum
	 */
	void set_parser_type(E_VW_PARSER_TYPE type = T_VW);

	/**
	 * Returns the parsed example.
	 *
	 * The example contains the label if available, and
	 * also contains length of the feature vector.
	 * These parameters are redundant.
	 *
	 * @param ex examples as VwExample*, set by reference
	 * @param len length of vector, untouched
	 */
	virtual void get_vector(VwExample* &ex, int32_t &len);

	/**
	 * Returns the parsed example.
	 *
	 * TODO: Make this fail if examples are found to be unlabelled.
	 *
	 * @param ex example as VwExample*, set by reference
	 * @param len length of vector, untouched
	 * @param label label, untouched
	 */
	virtual void get_vector_and_label(VwExample* &ex, int32_t &len, float64_t &label);

	/**
	 * Set environment for vw
	 *
	 * @param env_to_use CVwEnvironment* environment
	 */
	void set_env(CVwEnvironment* env_to_use)
	{
		parser->set_env(env_to_use);
	}

	/**
	 * Return the environment
	 *
	 * @return environment as CVwEnvironment*
	 */
	CVwEnvironment* get_env()
	{
		SG_REF(env);
		return env;
	}

	/**
	 * Set whether cache will be written
	 *
	 * @param write_cache whether to write to cache
	 */
	void set_write_to_cache(bool write_cache)
	{
		write_to_cache = write_cache;
		parser->set_write_cache(write_cache);
	}

	/**
	 * Get whether cache will be written
	 *
	 * @return whether to write to cache
	 */
	bool get_write_to_cache() { return write_to_cache; }

	virtual bool is_seekable() { return false; }

	/** @return object name */
	virtual const char* get_name() const
	{
		return "StreamingVwFile";
	}

public:
	/// The function which will be called for parsing
	parse_func parse_example;

private:
	/**
	 * Initialize members
	 */
	virtual void init();

protected:
	/// Parser for vw format
	CVwParser* parser;

	/// Parser type
	E_VW_PARSER_TYPE parser_type;

	/// Environment used for vw - used by parser
	CVwEnvironment* env;

	/// Write data to a binary cache file
	bool write_to_cache;
};
}
#endif //__STREAMING_VWFILE_H__
