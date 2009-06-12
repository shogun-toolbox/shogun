/*
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
 */

#ifndef HASH_H
#define HASH_H

#include <stdint.h>

#define MD5_DIGEST_LENGTH 16

class CHash : public CSGObject
{
	public:
		CHash() {}
		virtual ~CHash {}

		struct MD5Context {
			uint32_t buf[4];
			uint32_t bits[2];
			unsigned char in[64];
		};

		/// crc32
		static uint32_t crc32(uint8_t *data, int32_t len);

		/* Functions */
		static void MD5(unsigned char *x, unsigned l, unsigned char *buf);
		static void MD5Init(struct MD5Context *context);
		static void MD5Update(struct MD5Context *context,
				unsigned char const *buf, unsigned len);
		static void MD5Final(unsigned char digest[16],
				struct MD5Context *context);
		static void MD5Transform(uint32_t buf[4], uint32_t const in[16]);

		/** @return object name */
		inline virtual const char* get_name() const { return "Hash"; }
};
#endif
