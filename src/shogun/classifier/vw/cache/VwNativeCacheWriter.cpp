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

#include <classifier/vw/cache/VwNativeCacheWriter.h>

using namespace shogun;

CVwNativeCacheWriter::CVwNativeCacheWriter()
	: CVwCacheWriter()
{
	init();
}

CVwNativeCacheWriter::CVwNativeCacheWriter(char * fname, CVwEnvironment* env_to_use)
	: CVwCacheWriter(fname, env_to_use)
{
	init();
	buf.use_file(fd);

	write_header();
}

CVwNativeCacheWriter::~CVwNativeCacheWriter()
{
	buf.flush();
	buf.close_file();
}

void CVwNativeCacheWriter::set_file(int32_t f)
{
	if (fd > 0)
	{
		buf.flush();
		buf.close_file();
	}

	fd = f;
	buf.use_file(fd);

	write_header();
}

void CVwNativeCacheWriter::init()
{
	neg_1 = 1;
	general = 2;
	int_size = 6;
}

void CVwNativeCacheWriter::write_header()
{
	const char* vw_version = env->vw_version;
	vw_size_t numbits = env->num_bits;
	vw_size_t v_length = 4;

	// Version and numbits info
	buf.write_file(&v_length, sizeof(vw_size_t));
	buf.write_file(vw_version,v_length);
	buf.write_file(&numbits, sizeof(vw_size_t));
}

char* CVwNativeCacheWriter::run_len_encode(char *p, vw_size_t i)
{
	while (i >= 128)
	{
		*(p++) = (i & 127) | 128;
		i = i >> 7;
	}
	*(p++) = (i & 127);

	return p;
}

char* CVwNativeCacheWriter::bufcache_label(VwLabel* ld, char* c)
{
	*(float32_t*)c = ld->label;
	c += sizeof(ld->label);
	*(float32_t*)c = ld->weight;
	c += sizeof(ld->weight);
	*(float32_t*)c = ld->initial;
	c += sizeof(ld->initial);
	return c;
}

void CVwNativeCacheWriter::cache_label(VwLabel* ld)
{
	char *c;
	buf.buf_write(c, sizeof(ld->label)+sizeof(ld->weight)+sizeof(ld->initial));
	c = bufcache_label(ld,c);
}

void CVwNativeCacheWriter::cache_tag(v_array<char> tag)
{
	// Store the size of the tag and the tag itself
	char *c;

	buf.buf_write(c, sizeof(vw_size_t)+tag.index());
	*(vw_size_t*)c = tag.index();
	c += sizeof(vw_size_t);
	memcpy(c, tag.begin, tag.index());
	c += tag.index();

	buf.set(c);
}

void CVwNativeCacheWriter::output_byte(unsigned char s)
{
	char *c;

	buf.buf_write(c, 1);
	*(c++) = s;
	buf.set(c);
}

void CVwNativeCacheWriter::output_features(unsigned char index, VwFeature* begin, VwFeature* end)
{
	char* c;
	vw_size_t storage = (end-begin) * int_size;
	for (VwFeature* i = begin; i != end; i++)
		if (i->x != 1. && i->x != -1.)
			storage+=sizeof(float32_t);

	buf.buf_write(c, sizeof(index) + storage + sizeof(vw_size_t));
	*(unsigned char*)c = index;
	c += sizeof(index);

	char *storage_size_loc = c;
	c += sizeof(vw_size_t);

	vw_size_t last = 0;

	// Store the differences in hashed feature indices
	for (VwFeature* i = begin; i != end; i++)
	{
		int32_t s_diff = (i->weight_index - last);
		vw_size_t diff = ZigZagEncode(s_diff) << 2;
		last = i->weight_index;

		if (i->x == 1.)
			c = run_len_encode(c, diff);
		else if (i->x == -1.)
			c = run_len_encode(c, diff | neg_1);
		else
		{
			c = run_len_encode(c, diff | general);
			*(float32_t*)c = i->x;
			c += sizeof(float32_t);
		}
	}
	buf.set(c);
	*(vw_size_t*)storage_size_loc = c - storage_size_loc - sizeof(vw_size_t);
}

void CVwNativeCacheWriter::cache_example(VwExample* &ex)
{
	cache_label(ex->ld);
	cache_tag(ex->tag);
	output_byte(ex->indices.index());
	for (vw_size_t* b = ex->indices.begin; b != ex->indices.end; b++)
		output_features(*b, ex->atomics[*b].begin,ex->atomics[*b].end);
}

