/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/mathematics/Random.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CRandom::CRandom()
 : m_seed(12345)
{
	init();
}

CRandom::CRandom(uint32_t seed)
 : m_seed(seed)
{
	init();
}

CRandom::~CRandom()
{
#ifdef HAVE_PTHREAD
	PTHREAD_LOCK_DESTROY(&m_state_lock);
#endif
	SG_FREE(m_sfmt);
}

void CRandom::seed(uint32_t seed)
{
	m_seed = seed;
	reinit(PRNG_32);
}

void CRandom::init()
{
	m_sfmt = SG_MALLOC(sfmt_t, 1);
#ifdef HAVE_PTHREAD	
	PTHREAD_LOCK_INIT(&m_state_lock);
#endif	
	reinit(PRNG_32);
}

uint32_t CRandom::random_32()
{
	if (m_state != PRNG_32)
		reinit(PRNG_32);

	return sfmt_genrand_uint32(m_sfmt);
}

uint64_t CRandom::random_64()
{
	if (m_state != PRNG_64)
		reinit(PRNG_64);

	return sfmt_genrand_uint64(m_sfmt);
}

void CRandom::fill_array_32(uint32_t* array, int size)
{
	if (m_state != PRNG_32)
		reinit(PRNG_32);

	sfmt_fill_array32(m_sfmt, array, size);
}

void CRandom::fill_array_64(uint64_t* array, int size)
{
	if (m_state != PRNG_64)
		reinit(PRNG_64);

	sfmt_fill_array64(m_sfmt, array, size);
}

float64_t CRandom::random_close()
{
	if (m_state != PRNG_32)
		reinit(PRNG_32);

	return sfmt_genrand_real1(m_sfmt);
}

float64_t CRandom::random_open()
{
	if (m_state != PRNG_32)
		reinit(PRNG_32);

	return sfmt_genrand_real3(m_sfmt);
}

float64_t CRandom::random_half_open()
{
	if (m_state != PRNG_32)
		reinit(PRNG_32);

	return sfmt_genrand_real2(m_sfmt);
}

float64_t CRandom::random_float_res53()
{
	if (m_state != PRNG_64)
		reinit(PRNG_64);

	return sfmt_genrand_res53(m_sfmt);
}

void CRandom::reinit(PRNG_STATE state)
{
#ifdef HAVE_PTHREAD	
	PTHREAD_LOCK(&m_state_lock);
#endif	
	sfmt_init_gen_rand(m_sfmt, m_seed);
	m_state = state;
#ifdef HAVE_PTHREAD	
	PTHREAD_UNLOCK(&m_state_lock);
#endif	
}
