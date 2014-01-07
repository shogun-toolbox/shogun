/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#include <io/streaming/StreamingVwCacheFile.h>

using namespace shogun;

CStreamingVwCacheFile::CStreamingVwCacheFile()
	: CStreamingFile()
{
	buf=NULL;
	init(C_NATIVE);
}

CStreamingVwCacheFile::CStreamingVwCacheFile(EVwCacheType cache_type)
	: CStreamingFile()
{
	init(cache_type);
}

CStreamingVwCacheFile::CStreamingVwCacheFile(char* fname, char rw, EVwCacheType cache_type)
	: CStreamingFile(fname, rw)
{
	init(cache_type);
}

CStreamingVwCacheFile::~CStreamingVwCacheFile()
{
	SG_UNREF(env);
	SG_UNREF(cache_reader);
}

void CStreamingVwCacheFile::get_vector(VwExample* &ex, int32_t& len)
{
	if (cache_reader->read_cached_example(ex))
		len = 1;
	else
		len = -1;
}

void CStreamingVwCacheFile::get_vector_and_label(VwExample* &ex, int32_t &len, float64_t &label)
{
	if (cache_reader->read_cached_example(ex))
		len = 1;
	else
		len = -1;
}

void CStreamingVwCacheFile::set_env(CVwEnvironment* env_to_use)
{
	SG_REF(env_to_use);
	SG_UNREF(env);
	env = env_to_use;

	SG_UNREF(cache_reader);

	switch (cache_format)
	{
	case C_NATIVE:
		cache_reader = new CVwNativeCacheReader(buf->working_file, env);
		return;
	case C_PROTOBUF:
		SG_ERROR("Protocol buffers cache support is not implemented yet!\n")
	}

	SG_ERROR("Unexpected cache type to use for reading!\n")
}

void CStreamingVwCacheFile::reset_stream()
{
	buf->reset_file();

	// Recheck the cache so the parser can directly proceed with the examples
	if (cache_format == C_NATIVE)
		((CVwNativeCacheReader*) cache_reader)->check_cache_metadata();
}

void CStreamingVwCacheFile::init(EVwCacheType cache_type)
{
	cache_format = cache_type;
	env = new CVwEnvironment();

	switch (cache_type)
	{
	case C_NATIVE:
		if (buf)
			cache_reader = new CVwNativeCacheReader(buf->working_file, env);
		else
			cache_reader=NULL;
		return;
	case C_PROTOBUF:
		SG_ERROR("Protocol buffers cache support is not implemented yet!\n")
	}

	SG_ERROR("Unrecognized cache type to read from!\n")
}
