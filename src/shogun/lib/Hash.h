/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * The MD5 and Murmur hashing functions were integrated from public sources.
 * Their respective copyrights follow.
 *
 * MD5
 *
 * This code implements the MD5 message-digest algorithm.
 * The algorithm is due to Ron Rivest.  This code was
 * written by Colin Plumb in 1993, no copyright is claimed.
 * This code is in the public domain; do with it what you wish.
 *
 * Equivalent code is available from RSA Data Security, Inc.
 * This code has been tested against that, and is equivalent,
 * except that you don't need to include two pages of legalese
 * with every copy.
 *
 * To compute the message digest of a chunk of bytes, declare an
 * MD5Context structure, pass it to MD5Init, call MD5Update as
 * needed on buffers full of bytes, and then call MD5Final, which
 * will fill a supplied 16-byte array with the digest.
 *
 * MurmurHash2
 *
 * (C) Austin Appleby, released under the MIT License
 *
 *  Note - This code makes a few assumptions about how your machine behaves -
 *
 *  1. We can read a 4-byte value from any address without crashing
 *  2. It will not produce the same results on little-endian and big-endian
 *     machines.
 */

#ifndef HASH_H
#define HASH_H

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{
/** @brief Collection of Hashing Functions
 *
 * This class implements a number of hashing functions like
 * crc32, md5 and murmur.
 *
 */
class CHash : public CSGObject
{
	public:
		/** default constructor */
		CHash() {}
		/** default destructor */
		virtual ~CHash() {}

		/** crc32 checksumming
		 *
		 * @param data data to checksum
		 * @param len length in number of bytes
		 */
		static uint32_t crc32(uint8_t *data, int32_t len);

		/** Wrapper for MD5 function compatible to OpenSSL interface.
		 *
		 * @param x data
		 * @param l length
		 * @param buf buf needs to provide an allocated array of 16 bytes
		 *        for the digest.
		 */
		static void MD5(unsigned char *x, unsigned l, unsigned char *buf);

		/** Murmur Hash2
		 *
		 * @param data data to checksum (needs to be 32bit aligned on some archs)
		 * @param len length in number of bytes
		 * @param seed initial seed
		 *
		 * @return hash
		 */
		static uint32_t MurmurHash2(uint8_t* data, int32_t len, uint32_t seed);

		/** Incremental Murmur like Hash
		 *
		 * @param data byte to hash
		 * @param h initial seed / hash on subsequent calls
		 *
		 * @return hash
		 */
		static uint32_t IncrementalMurmurHash2(uint8_t data, uint32_t h);

		/** Apply Murmur Hash on the non-numeric part of
		 * a substring.
		 *
		 * The integral part is returned as-is.
		 *
		 * @param s substring
		 * @param h initial seed
		 *
		 * @return hash
		 */
		static uint32_t MurmurHashString(substring s, uint32_t h);

		/** @return object name */
		inline virtual const char* get_name() const { return "Hash"; }

	protected:

#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** MD5Context */
		struct MD5Context {
			/** 16 byte buffer */
			uint32_t buf[4];
			/** 8 byte buffer */
			uint32_t bits[2];
			/** 64 byte buffer */
			unsigned char in[64];
		};
#endif // DOXYGEN_SHOULD_SKIP_THIS

		/**
		 * Start MD5 accumulation.  Set bit count to 0 and buffer to mysterious
		 * initialization constants.
		 *
		 * @param context MD5Context
		 */
		static void MD5Init(struct MD5Context *context);

		/**
		 * Update context to reflect the concatenation of another buffer full
		 * of bytes.
		 *
		 * @param context initialized MD5Context
		 * @param buf buffer to hash
		 * @param len length of buffer
		 */
		static void MD5Update(struct MD5Context *context,
				unsigned char const *buf, unsigned len);

		/**
		 * Final wrapup - pad to 64-byte boundary with the bit pattern
		 * 1 0* (64-bit count of bits processed, MSB-first)
		 *
		 * @param digest the 64 byte hash
		 * @param context initialized MD5Context
		 */
		static void MD5Final(unsigned char digest[16],
				struct MD5Context *context);
		/**
		 * The core of the MD5 algorithm, this alters an existing MD5 hash to
		 * reflect the addition of 16 longwords of new data.  MD5Update blocks
		 * the data and converts bytes into longwords for this routine.
		 *
		 * @param buf 16 byte
		 * @param in 64 bytes
		 */
		static void MD5Transform(uint32_t buf[4], uint32_t const in[16]);
};
}
#endif
