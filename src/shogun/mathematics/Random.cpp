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
	SG_FREE(m_sfmt_32);
	SG_FREE(m_sfmt_64);
}

void CRandom::set_seed(uint32_t seed)
{	
	reinit(seed);
}

uint32_t CRandom::get_seed() const
{
	return m_seed;
}

void CRandom::init()
{
	m_sfmt_32 = SG_MALLOC(sfmt_t, 1);
	m_sfmt_64 = SG_MALLOC(sfmt_t, 1);
#ifdef HAVE_PTHREAD	
	PTHREAD_LOCK_INIT(&m_state_lock);
#endif
	reinit(m_seed);
}

uint32_t CRandom::random_32()
{
	return sfmt_genrand_uint32(m_sfmt_32);
}

uint64_t CRandom::random_64()
{
	return sfmt_genrand_uint64(m_sfmt_64);
}

void CRandom::fill_array_32(uint32_t* array, int size)
{
	sfmt_fill_array32(m_sfmt_32, array, size);
}

void CRandom::fill_array_64(uint64_t* array, int size)
{
	sfmt_fill_array64(m_sfmt_64, array, size);
}

float64_t CRandom::random_close()
{
	return sfmt_genrand_real1(m_sfmt_32);
}

float64_t CRandom::random_open()
{
	return sfmt_genrand_real3(m_sfmt_32);
}

float64_t CRandom::random_half_open()
{
	return sfmt_genrand_real2(m_sfmt_32);
}

float64_t CRandom::random_float_res53()
{
	return sfmt_genrand_res53(m_sfmt_64);
}
