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
// Tweaked for VW and contributed by Ariel Faigon.
// Original at: http://murmurhash.googlepages.com/
//
// Based on MurmurHash2, by Austin Appleby
//
// Note - This code makes a few assumptions about how your machine behaves:
//
// 1. We can read a 4-byte value from any address without crashing
//    (i.e non aligned access is supported) - this is not a problem
//    on Intel/x86/AMD64 machines (including new Macs)
// 2. sizeof(int) == 4
//
// And it has a few limitations -
//  1. It will not work incrementally.
//  2. It will not produce the same results on little-endian and
//     big-endian machines.
//

#include <shogun/classifier/vw/hash.h>

#include <stdint.h>
#include <sys/types.h>

#define MIX(h,k,m) { k *= m; k ^= k >> r; k *= m; h *= m; h ^= k; }

namespace shogun
{
size_t hashstring (substring s, unsigned long h)
{
	size_t ret = 0;
	//trim leading whitespace
	for(; *(s.start) <= 0x20 && s.start < s.end; s.start++);
	//trim trailing white space
	for(; *(s.end-1) <= 0x20 && s.end > s.start; s.end--);

	char *p = s.start;
	while (p != s.end)
		if (isdigit(*p))
			ret = 10*ret + *(p++) - '0';
		else
			return uniform_hash((unsigned char *)s.start, s.end - s.start, h);

	return ret + h;
}

size_t hashall (substring s, unsigned long h)
{
	return uniform_hash((unsigned char *)s.start, s.end - s.start, h);
}

hash_func_t getHasher(char*& s)
{
	if (strcmp(s,"strings") == 0)
		return hashstring;
	else if(strcmp(s, "all") == 0)
		return hashall;
	else
		SG_SERROR("Unknown hash function: %s\n");
	return NULL;
}

uint32_t uniform_hash(const void *key, size_t len, uint32_t seed)
{
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.

	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	// Initialize the hash to a 'random' value

	unsigned int h = seed ^ len;

	// Mix 4 bytes at a time into the hash

	const unsigned char * data = (const unsigned char *)key;

	while (len >= 4)
	{
		unsigned int k = *(unsigned int *)data;

		k *= m;
		k ^= k >> r;
		k *= m;

		h *= m;
		h ^= k;

		data += 4;
		len -= 4;
	}

	// Handle the last few bytes of the input array
	switch (len)
	{
	case 3: h ^= data[2] << 16;
	case 2: h ^= data[1] << 8;
	case 1: h ^= data[0];
		h *= m;
	};

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.
	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;

	return h;
}
}
