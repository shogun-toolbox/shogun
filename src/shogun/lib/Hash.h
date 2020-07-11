/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Jacob Walker, Yuyu Zhang, Shashwat Lal Das, 
 *          Thoralf Klein, Evan Shelhamer
 */

#ifndef HASH_H
#define HASH_H

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/common.h>

namespace shogun
{
struct substring;

/** @brief Collection of Hashing Functions
 *
 * This class implements a number of hashing functions like
 * crc32, md5 and murmur.
 *
 */
class Hash : public SGObject
{
	public:
		/** default constructor */
		Hash() {}
		/** default destructor */
		~Hash() override {}

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

		/** Murmur Hash3
		 * Wrapper for function in PMurHash.c
		 *
		 * @param data data to checksum (needs to be 32bit aligned on some archs)
		 * @param len length in number of bytes
		 * @param seed initial seed
		 *
		 * @return hash
		 */
		static uint32_t MurmurHash3(uint8_t* data, int32_t len, uint32_t seed);

		/** Incremental Murmur3 Hash. Wrapper for function in PMurHash.c
		 * FinalizeIncrementalMurmurHash3 must be called
		 * at the end of all incremental hashing to
		 * obtain final hash.
		 *
		 * @param hash value. (The value returned is not the final hash).
		 * @param carry value for incremental hash. See PMurHash.c for details.
		 * @param data data to checksum (needs to be 32bit aligned on some archs)
		 * @param len length in number of bytes
		 *
		 */
		static void IncrementalMurmurHash3(uint32_t *hash, uint32_t *carry,
				uint8_t* data, int32_t len);

		/** Finalize Incremental Murmur Hash. Called when all
		 * incremental hashing is done to get final hash value.
		 *
		 * Wrapper for function in PMurHash.c
		 *
		 * @param h hash returned by IncrementalMurmurHash3
		 * @param carry returned by IncrementalMurmurHash3
		 * @param total_length total length of all bytes hashed.
		 *
		 */
		static uint32_t FinalizeIncrementalMurmurHash3(uint32_t h,
				uint32_t carry, uint32_t total_length);

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
		const char* get_name() const override { return "Hash"; }

	protected:

#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** MD5Context */
		struct MD5Context {
			/** 16 byte buffer */
			uint32_t buf[4];
			/** 8 byte buffer */
			uint32_t bits[2];
			union
			{
				/** 64 byte buffer */
				unsigned char in[64];
				/** and equivalently 16 uint32's */
				uint32_t uin[16];
			};
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
