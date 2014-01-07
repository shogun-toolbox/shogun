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

#include <classifier/vw/cache/VwNativeCacheReader.h>

using namespace shogun;

CVwNativeCacheReader::CVwNativeCacheReader()
	: CVwCacheReader(), char_size(2)
{
	init();
}

CVwNativeCacheReader::CVwNativeCacheReader(char * fname, CVwEnvironment* env_to_use)
	: CVwCacheReader(fname, env_to_use), char_size(2)
{
	init();
	buf.use_file(fd);
	check_cache_metadata();
}

CVwNativeCacheReader::CVwNativeCacheReader(int32_t f, CVwEnvironment* env_to_use)
	: CVwCacheReader(f, env_to_use), char_size(2)
{
	init();
	buf.use_file(fd);
	check_cache_metadata();
}

CVwNativeCacheReader::~CVwNativeCacheReader()
{
	buf.close_file();
}

void CVwNativeCacheReader::set_file(int32_t f)
{
	if (fd > 0)
		buf.close_file();

	fd = f;
	buf.use_file(fd);
	check_cache_metadata();
}

void CVwNativeCacheReader::init()
{
	neg_1 = 1;
	general = 2;
}

void CVwNativeCacheReader::check_cache_metadata()
{
	const char* vw_version=env->vw_version;
	vw_size_t numbits = env->num_bits;

	vw_size_t v_length;
	buf.read_file((char*)&v_length, sizeof(v_length));
	if(v_length > 29)
		SG_SERROR("Cache version too long, cache file is probably invalid.\n")

	char* t=SG_MALLOC(char, v_length);
	buf.read_file(t,v_length);
	if (strcmp(t,vw_version) != 0)
	{
		SG_FREE(t);
		SG_SERROR("Cache has possibly incompatible version!\n")
	}
	SG_FREE(t);

	vw_size_t cache_numbits = 0;
	if (buf.read_file(&cache_numbits, sizeof(vw_size_t)) < ssize_t(sizeof(vw_size_t)))
		return;

	if (cache_numbits != numbits)
		SG_SERROR("Bug encountered in caching! Bits used for weight in cache: %d.\n", cache_numbits)
}

char* CVwNativeCacheReader::run_len_decode(char *p, vw_size_t& i)
{
	// Read an int32_t 7 bits at a time.
	vw_size_t count = 0;
	while(*p & 128)\
		i = i | ((*(p++) & 127) << 7*count++);
	i = i | (*(p++) << 7*count);
	return p;
}

char* CVwNativeCacheReader::bufread_label(VwLabel* const ld, char* c)
{
	ld->label = *(float32_t*)c;
	c += sizeof(ld->label);
	set_minmax(ld->label);

	ld->weight = *(float32_t*)c;
	c += sizeof(ld->weight);
	ld->initial = *(float32_t*)c;
	c += sizeof(ld->initial);

	return c;
}

vw_size_t CVwNativeCacheReader::read_cached_label(VwLabel* const ld)
{
	char *c;
	vw_size_t total = sizeof(ld->label)+sizeof(ld->weight)+sizeof(ld->initial);
	if (buf.buf_read(c, total) < total)
		return 0;
	c = bufread_label(ld,c);

	return total;
}

vw_size_t CVwNativeCacheReader::read_cached_tag(VwExample* const ae)
{
	char* c;
	vw_size_t tag_size;
	if (buf.buf_read(c, sizeof(tag_size)) < sizeof(tag_size))
		return 0;
	tag_size = *(vw_size_t*)c;
	c += sizeof(tag_size);

	buf.set(c);
	if (buf.buf_read(c, tag_size) < tag_size)
		return 0;

	ae->tag.erase();
	ae->tag.push_many(c, tag_size);
	return tag_size+sizeof(tag_size);
}

bool CVwNativeCacheReader::read_cached_example(VwExample* const ae)
{
	vw_size_t mask =  env->mask;
	vw_size_t total = read_cached_label(ae->ld);
	if (total == 0)
		return false;
	if (read_cached_tag(ae) == 0)
		return false;

	char* c;
	unsigned char num_indices = 0;
	if (buf.buf_read(c, sizeof(num_indices)) < sizeof(num_indices))
		return false;
	num_indices = *(unsigned char*)c;
	c += sizeof(num_indices);

	buf.set(c);

	for (; num_indices > 0; num_indices--)
	{
		vw_size_t temp;
		unsigned char index = 0;
		temp = buf.buf_read(c, sizeof(index) + sizeof(vw_size_t));

		if (temp < sizeof(index) + sizeof(vw_size_t))
			SG_SERROR("Truncated example! %d < %d bytes expected.\n",
				  temp, char_size + sizeof(vw_size_t));

		index = *(unsigned char*) c;
		c += sizeof(index);
		ae->indices.push((vw_size_t) index);

		v_array<VwFeature>* ours = ae->atomics+index;
		float64_t* our_sum_feat_sq = ae->sum_feat_sq+index;
		vw_size_t storage = *(vw_size_t *)c;
		c += sizeof(vw_size_t);

		buf.set(c);
		total += storage;
		if (buf.buf_read(c, storage) < storage)
			SG_SERROR("Truncated example! Wanted %d bytes!\n", storage)

		char *end = c + storage;

		vw_size_t last = 0;

		for (; c!=end; )
		{
			VwFeature f = {1., 0};
			temp = f.weight_index;
			c = run_len_decode(c, temp);
			f.weight_index = temp;

			if (f.weight_index & neg_1)
				f.x = -1.;
			else if (f.weight_index & general)
			{
				f.x = ((one_float*)c)->f;
				c += sizeof(float32_t);
			}

			*our_sum_feat_sq += f.x*f.x;

			vw_size_t diff = f.weight_index >> 2;
			int32_t s_diff = ZigZagDecode(diff);
			if (s_diff < 0)
				ae->sorted = false;

			f.weight_index = last + s_diff;
			last = f.weight_index;
			f.weight_index = f.weight_index & mask;

			ours->push(f);
		}
		buf.set(c);
	}

	return true;
}
