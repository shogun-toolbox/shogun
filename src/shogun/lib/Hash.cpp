/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 *
 * The MD5 hashing function was integrated from public sources.
 * Its copyright follows.
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
 */

#include <lib/common.h>
#include <lib/Hash.h>
#include <ctype.h>

using namespace shogun;

uint32_t CHash::crc32(uint8_t *data, int32_t len)
{
	uint32_t result;
	int32_t i,j;
	uint8_t octet;

	result = 0-1;
	for (i=0; i<len; i++)
	{
		octet = *(data++);
		for (j=0; j<8; j++)
		{
			if ((octet >> 7) ^ (result >> 31))
			{
				result = (result << 1) ^ 0x04c11db7;
			}
			else
			{
				result = (result << 1);
			}
			octet <<= 1;
		}
	}

    return ~result;
}

void CHash::MD5(unsigned char *x, unsigned l, unsigned char *buf)
{
    struct MD5Context ctx;

    MD5Init(&ctx);
    MD5Update(&ctx, x, l);
    MD5Final(buf, &ctx);
}

#ifndef HIGHFIRST
#define byteReverse(buf, len)   /* Nothing */
#else
void byteReverse(unsigned char *buf, unsigned uint32_t longs);

#ifndef ASM_MD5
/*
 * Note: this code is harmless on little-endian machines.
 */
void byteReverse(unsigned char *buf, unsigned uint32_t longs)
{
    uint32_t t;
    do {
        t = (uint32_t) ((unsigned) buf[3] << 8 | buf[2]) << 16 |
            ((unsigned) buf[1] << 8 | buf[0]);
        *(uint32_t *) buf = t;
        buf += 4;
    } while (--longs);
}
#endif
#endif

void CHash::MD5Init(struct MD5Context *ctx)
{
    ctx->buf[0] = 0x67452301;
    ctx->buf[1] = 0xefcdab89;
    ctx->buf[2] = 0x98badcfe;
    ctx->buf[3] = 0x10325476;

    ctx->bits[0] = 0;
    ctx->bits[1] = 0;
}

void CHash::MD5Update(struct MD5Context *ctx, unsigned char const *buf,
               unsigned len)
{
    uint32_t t;

    /* Update bitcount */

    t = ctx->bits[0];
    if ((ctx->bits[0] = t + ((uint32_t) len << 3)) < t)
        ctx->bits[1]++;         /* Carry from low to high */
    ctx->bits[1] += len >> 29;

    t = (t >> 3) & 0x3f;        /* Bytes already in shsInfo->data */

    /* Handle any leading odd-sized chunks */

    if (t) {
        unsigned char *p = (unsigned char *) ctx->in + t;

        t = 64 - t;
        if (len < t) {
            memcpy(p, buf, len);
            return;
        }
        memcpy(p, buf, t);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (uint32_t *) ctx->in);
        buf += t;
        len -= t;
    }
    /* Process data in 64-byte chunks */

    while (len >= 64) {
        memcpy(ctx->in, buf, 64);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (uint32_t *) ctx->in);
        buf += 64;
        len -= 64;
    }

    /* Handle any remaining bytes of data. */

    memcpy(ctx->in, buf, len);
}

void CHash::MD5Final(unsigned char digest[16], struct MD5Context *ctx)
{
    unsigned count;
    unsigned char *p;

    /* Compute number of bytes mod 64 */
    count = (ctx->bits[0] >> 3) & 0x3F;

    /* Set the first char of padding to 0x80.  This is safe since there is
       always at least one byte free */
    p = ctx->in + count;
    *p++ = 0x80;

    /* Bytes of padding needed to make 64 bytes */
    count = 64 - 1 - count;

    /* Pad out to 56 mod 64 */
    if (count < 8) {
        /* Two lots of padding:  Pad the first block to 64 bytes */
        memset(p, 0, count);
        byteReverse(ctx->in, 16);
        MD5Transform(ctx->buf, (uint32_t *) ctx->in);

        /* Now fill the next block with 56 bytes */
        memset(ctx->in, 0, 56);
    } else {
        /* Pad block to 56 bytes */
        memset(p, 0, count - 8);
    }
    byteReverse(ctx->in, 14);

    /* Append length in bits and transform */
    ctx->uin[14] = ctx->bits[0];
    ctx->uin[15] = ctx->bits[1];

    MD5Transform(ctx->buf, (uint32_t *) ctx->in);
    byteReverse((unsigned char *) ctx->buf, 4);
    memcpy(digest, ctx->buf, 16);
    memset(ctx, 0, sizeof(*ctx));        /* In case it's sensitive */
}

#ifndef ASM_MD5

/* The four core functions - F1 is optimized somewhat */

/* #define F1(x, y, z) (x & y | ~x & z) */
#define F1(x, y, z) (z ^ (x & (y ^ z)))
#define F2(x, y, z) F1(z, x, y)
#define F3(x, y, z) (x ^ y ^ z)
#define F4(x, y, z) (y ^ (x | ~z))

/* This is the central step in the MD5 algorithm. */
#ifdef __PUREC__
#define MD5STEP(f, w, x, y, z, data, s) \
	( w += f /*(x, y, z)*/ + data,  w = w<<s | w>>(32-s),  w += x )
#else
#define MD5STEP(f, w, x, y, z, data, s) \
	( w += f(x, y, z) + data,  w = w<<s | w>>(32-s),  w += x )
#endif

void CHash::MD5Transform(uint32_t buf[4], uint32_t const in[16])
{
    register uint32_t a, b, c, d;

    a = buf[0];
    b = buf[1];
    c = buf[2];
    d = buf[3];

#ifdef __PUREC__                /* PureC Weirdness... (GG) */
    MD5STEP(F1(b, c, d), a, b, c, d, in[0] + 0xd76aa478L, 7);
    MD5STEP(F1(a, b, c), d, a, b, c, in[1] + 0xe8c7b756L, 12);
    MD5STEP(F1(d, a, b), c, d, a, b, in[2] + 0x242070dbL, 17);
    MD5STEP(F1(c, d, a), b, c, d, a, in[3] + 0xc1bdceeeL, 22);
    MD5STEP(F1(b, c, d), a, b, c, d, in[4] + 0xf57c0fafL, 7);
    MD5STEP(F1(a, b, c), d, a, b, c, in[5] + 0x4787c62aL, 12);
    MD5STEP(F1(d, a, b), c, d, a, b, in[6] + 0xa8304613L, 17);
    MD5STEP(F1(c, d, a), b, c, d, a, in[7] + 0xfd469501L, 22);
    MD5STEP(F1(b, c, d), a, b, c, d, in[8] + 0x698098d8L, 7);
    MD5STEP(F1(a, b, c), d, a, b, c, in[9] + 0x8b44f7afL, 12);
    MD5STEP(F1(d, a, b), c, d, a, b, in[10] + 0xffff5bb1L, 17);
    MD5STEP(F1(c, d, a), b, c, d, a, in[11] + 0x895cd7beL, 22);
    MD5STEP(F1(b, c, d), a, b, c, d, in[12] + 0x6b901122L, 7);
    MD5STEP(F1(a, b, c), d, a, b, c, in[13] + 0xfd987193L, 12);
    MD5STEP(F1(d, a, b), c, d, a, b, in[14] + 0xa679438eL, 17);
    MD5STEP(F1(c, d, a), b, c, d, a, in[15] + 0x49b40821L, 22);

    MD5STEP(F2(b, c, d), a, b, c, d, in[1] + 0xf61e2562L, 5);
    MD5STEP(F2(a, b, c), d, a, b, c, in[6] + 0xc040b340L, 9);
    MD5STEP(F2(d, a, b), c, d, a, b, in[11] + 0x265e5a51L, 14);
    MD5STEP(F2(c, d, a), b, c, d, a, in[0] + 0xe9b6c7aaL, 20);
    MD5STEP(F2(b, c, d), a, b, c, d, in[5] + 0xd62f105dL, 5);
    MD5STEP(F2(a, b, c), d, a, b, c, in[10] + 0x02441453L, 9);
    MD5STEP(F2(d, a, b), c, d, a, b, in[15] + 0xd8a1e681L, 14);
    MD5STEP(F2(c, d, a), b, c, d, a, in[4] + 0xe7d3fbc8L, 20);
    MD5STEP(F2(b, c, d), a, b, c, d, in[9] + 0x21e1cde6L, 5);
    MD5STEP(F2(a, b, c), d, a, b, c, in[14] + 0xc33707d6L, 9);
    MD5STEP(F2(d, a, b), c, d, a, b, in[3] + 0xf4d50d87L, 14);
    MD5STEP(F2(c, d, a), b, c, d, a, in[8] + 0x455a14edL, 20);
    MD5STEP(F2(b, c, d), a, b, c, d, in[13] + 0xa9e3e905L, 5);
    MD5STEP(F2(a, b, c), d, a, b, c, in[2] + 0xfcefa3f8L, 9);
    MD5STEP(F2(d, a, b), c, d, a, b, in[7] + 0x676f02d9L, 14);
    MD5STEP(F2(c, d, a), b, c, d, a, in[12] + 0x8d2a4c8aL, 20);

    MD5STEP(F3(b, c, d), a, b, c, d, in[5] + 0xfffa3942L, 4);
    MD5STEP(F3(a, b, c), d, a, b, c, in[8] + 0x8771f681L, 11);
    MD5STEP(F3(d, a, b), c, d, a, b, in[11] + 0x6d9d6122L, 16);
    MD5STEP(F3(c, d, a), b, c, d, a, in[14] + 0xfde5380cL, 23);
    MD5STEP(F3(b, c, d), a, b, c, d, in[1] + 0xa4beea44L, 4);
    MD5STEP(F3(a, b, c), d, a, b, c, in[4] + 0x4bdecfa9L, 11);
    MD5STEP(F3(d, a, b), c, d, a, b, in[7] + 0xf6bb4b60L, 16);
    MD5STEP(F3(c, d, a), b, c, d, a, in[10] + 0xbebfbc70L, 23);
    MD5STEP(F3(b, c, d), a, b, c, d, in[13] + 0x289b7ec6L, 4);
    MD5STEP(F3(a, b, c), d, a, b, c, in[0] + 0xeaa127faL, 11);
    MD5STEP(F3(d, a, b), c, d, a, b, in[3] + 0xd4ef3085L, 16);
    MD5STEP(F3(c, d, a), b, c, d, a, in[6] + 0x04881d05L, 23);
    MD5STEP(F3(b, c, d), a, b, c, d, in[9] + 0xd9d4d039L, 4);
    MD5STEP(F3(a, b, c), d, a, b, c, in[12] + 0xe6db99e5L, 11);
    MD5STEP(F3(d, a, b), c, d, a, b, in[15] + 0x1fa27cf8L, 16);
    MD5STEP(F3(c, d, a), b, c, d, a, in[2] + 0xc4ac5665L, 23);

    MD5STEP(F4(b, c, d), a, b, c, d, in[0] + 0xf4292244L, 6);
    MD5STEP(F4(a, b, c), d, a, b, c, in[7] + 0x432aff97L, 10);
    MD5STEP(F4(d, a, b), c, d, a, b, in[14] + 0xab9423a7L, 15);
    MD5STEP(F4(c, d, a), b, c, d, a, in[5] + 0xfc93a039L, 21);
    MD5STEP(F4(b, c, d), a, b, c, d, in[12] + 0x655b59c3L, 6);
    MD5STEP(F4(a, b, c), d, a, b, c, in[3] + 0x8f0ccc92L, 10);
    MD5STEP(F4(d, a, b), c, d, a, b, in[10] + 0xffeff47dL, 15);
    MD5STEP(F4(c, d, a), b, c, d, a, in[1] + 0x85845dd1L, 21);
    MD5STEP(F4(b, c, d), a, b, c, d, in[8] + 0x6fa87e4fL, 6);
    MD5STEP(F4(a, b, c), d, a, b, c, in[15] + 0xfe2ce6e0L, 10);
    MD5STEP(F4(d, a, b), c, d, a, b, in[6] + 0xa3014314L, 15);
    MD5STEP(F4(c, d, a), b, c, d, a, in[13] + 0x4e0811a1L, 21);
    MD5STEP(F4(b, c, d), a, b, c, d, in[4] + 0xf7537e82L, 6);
    MD5STEP(F4(a, b, c), d, a, b, c, in[11] + 0xbd3af235L, 10);
    MD5STEP(F4(d, a, b), c, d, a, b, in[2] + 0x2ad7d2bbL, 15);
    MD5STEP(F4(c, d, a), b, c, d, a, in[9] + 0xeb86d391L, 21);
#else
    MD5STEP(F1, a, b, c, d, in[0] + 0xd76aa478, 7);
    MD5STEP(F1, d, a, b, c, in[1] + 0xe8c7b756, 12);
    MD5STEP(F1, c, d, a, b, in[2] + 0x242070db, 17);
    MD5STEP(F1, b, c, d, a, in[3] + 0xc1bdceee, 22);
    MD5STEP(F1, a, b, c, d, in[4] + 0xf57c0faf, 7);
    MD5STEP(F1, d, a, b, c, in[5] + 0x4787c62a, 12);
    MD5STEP(F1, c, d, a, b, in[6] + 0xa8304613, 17);
    MD5STEP(F1, b, c, d, a, in[7] + 0xfd469501, 22);
    MD5STEP(F1, a, b, c, d, in[8] + 0x698098d8, 7);
    MD5STEP(F1, d, a, b, c, in[9] + 0x8b44f7af, 12);
    MD5STEP(F1, c, d, a, b, in[10] + 0xffff5bb1, 17);
    MD5STEP(F1, b, c, d, a, in[11] + 0x895cd7be, 22);
    MD5STEP(F1, a, b, c, d, in[12] + 0x6b901122, 7);
    MD5STEP(F1, d, a, b, c, in[13] + 0xfd987193, 12);
    MD5STEP(F1, c, d, a, b, in[14] + 0xa679438e, 17);
    MD5STEP(F1, b, c, d, a, in[15] + 0x49b40821, 22);

    MD5STEP(F2, a, b, c, d, in[1] + 0xf61e2562, 5);
    MD5STEP(F2, d, a, b, c, in[6] + 0xc040b340, 9);
    MD5STEP(F2, c, d, a, b, in[11] + 0x265e5a51, 14);
    MD5STEP(F2, b, c, d, a, in[0] + 0xe9b6c7aa, 20);
    MD5STEP(F2, a, b, c, d, in[5] + 0xd62f105d, 5);
    MD5STEP(F2, d, a, b, c, in[10] + 0x02441453, 9);
    MD5STEP(F2, c, d, a, b, in[15] + 0xd8a1e681, 14);
    MD5STEP(F2, b, c, d, a, in[4] + 0xe7d3fbc8, 20);
    MD5STEP(F2, a, b, c, d, in[9] + 0x21e1cde6, 5);
    MD5STEP(F2, d, a, b, c, in[14] + 0xc33707d6, 9);
    MD5STEP(F2, c, d, a, b, in[3] + 0xf4d50d87, 14);
    MD5STEP(F2, b, c, d, a, in[8] + 0x455a14ed, 20);
    MD5STEP(F2, a, b, c, d, in[13] + 0xa9e3e905, 5);
    MD5STEP(F2, d, a, b, c, in[2] + 0xfcefa3f8, 9);
    MD5STEP(F2, c, d, a, b, in[7] + 0x676f02d9, 14);
    MD5STEP(F2, b, c, d, a, in[12] + 0x8d2a4c8a, 20);

    MD5STEP(F3, a, b, c, d, in[5] + 0xfffa3942, 4);
    MD5STEP(F3, d, a, b, c, in[8] + 0x8771f681, 11);
    MD5STEP(F3, c, d, a, b, in[11] + 0x6d9d6122, 16);
    MD5STEP(F3, b, c, d, a, in[14] + 0xfde5380c, 23);
    MD5STEP(F3, a, b, c, d, in[1] + 0xa4beea44, 4);
    MD5STEP(F3, d, a, b, c, in[4] + 0x4bdecfa9, 11);
    MD5STEP(F3, c, d, a, b, in[7] + 0xf6bb4b60, 16);
    MD5STEP(F3, b, c, d, a, in[10] + 0xbebfbc70, 23);
    MD5STEP(F3, a, b, c, d, in[13] + 0x289b7ec6, 4);
    MD5STEP(F3, d, a, b, c, in[0] + 0xeaa127fa, 11);
    MD5STEP(F3, c, d, a, b, in[3] + 0xd4ef3085, 16);
    MD5STEP(F3, b, c, d, a, in[6] + 0x04881d05, 23);
    MD5STEP(F3, a, b, c, d, in[9] + 0xd9d4d039, 4);
    MD5STEP(F3, d, a, b, c, in[12] + 0xe6db99e5, 11);
    MD5STEP(F3, c, d, a, b, in[15] + 0x1fa27cf8, 16);
    MD5STEP(F3, b, c, d, a, in[2] + 0xc4ac5665, 23);

    MD5STEP(F4, a, b, c, d, in[0] + 0xf4292244, 6);
    MD5STEP(F4, d, a, b, c, in[7] + 0x432aff97, 10);
    MD5STEP(F4, c, d, a, b, in[14] + 0xab9423a7, 15);
    MD5STEP(F4, b, c, d, a, in[5] + 0xfc93a039, 21);
    MD5STEP(F4, a, b, c, d, in[12] + 0x655b59c3, 6);
    MD5STEP(F4, d, a, b, c, in[3] + 0x8f0ccc92, 10);
    MD5STEP(F4, c, d, a, b, in[10] + 0xffeff47d, 15);
    MD5STEP(F4, b, c, d, a, in[1] + 0x85845dd1, 21);
    MD5STEP(F4, a, b, c, d, in[8] + 0x6fa87e4f, 6);
    MD5STEP(F4, d, a, b, c, in[15] + 0xfe2ce6e0, 10);
    MD5STEP(F4, c, d, a, b, in[6] + 0xa3014314, 15);
    MD5STEP(F4, b, c, d, a, in[13] + 0x4e0811a1, 21);
    MD5STEP(F4, a, b, c, d, in[4] + 0xf7537e82, 6);
    MD5STEP(F4, d, a, b, c, in[11] + 0xbd3af235, 10);
    MD5STEP(F4, c, d, a, b, in[2] + 0x2ad7d2bb, 15);
    MD5STEP(F4, b, c, d, a, in[9] + 0xeb86d391, 21);
#endif

    buf[0] += a;
    buf[1] += b;
    buf[2] += c;
    buf[3] += d;
}
#endif

uint32_t CHash::MurmurHash3(uint8_t* data, int32_t len, uint32_t seed)
{
	return PMurHash32(seed, data, len);
}

void CHash::IncrementalMurmurHash3(uint32_t *ph1, uint32_t *pcarry, uint8_t* data, int32_t len)
{
	PMurHash32_Process(ph1, pcarry, data, len);
}

uint32_t CHash::FinalizeIncrementalMurmurHash3(uint32_t h, uint32_t carry, uint32_t total_length)
{
	return PMurHash32_Result(h, carry, total_length);
}

uint32_t CHash::MurmurHashString(substring s, uint32_t h)
{
	uint32_t ret = 0;

	// Trim leading whitespace
	for(; *(s.start) <= 0x20 && s.start < s.end; s.start++);

	// Trim trailing white space
	for(; *(s.end-1) <= 0x20 && s.end > s.start; s.end--);

	char *p = s.start;
	while (p != s.end)
		if (isdigit(*p))
			ret = 10*ret + *(p++) - '0';
		else
			return MurmurHash3((uint8_t *)s.start, s.end - s.start, h);

	return ret + h;
}
