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
 : m_seed(12345),
 m_sfmt_32(NULL),
 m_sfmt_64(NULL),
 m_dsfmt(NULL)
{
	init();
}

CRandom::CRandom(uint32_t seed)
 : m_seed(seed),
 m_sfmt_32(NULL),
 m_sfmt_64(NULL),
 m_dsfmt(NULL)
{
	init();
}

CRandom::~CRandom()
{
	SG_FREE(m_sfmt_32);
	SG_FREE(m_sfmt_64);
	SG_FREE(m_dsfmt);
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
	m_dsfmt = SG_MALLOC(dsfmt_t, 1);
#ifdef HAVE_PTHREAD	
	PTHREAD_LOCK_INIT(&m_state_lock);
#endif
	reinit(m_seed);
}

uint32_t CRandom::random_32() const
{
	return sfmt_genrand_uint32(m_sfmt_32);
}

uint64_t CRandom::random_64() const
{
	return sfmt_genrand_uint64(m_sfmt_64);
}

void CRandom::fill_array(uint32_t* array, int32_t size) const
{
	if ((size >= sfmt_get_min_array_size32(m_sfmt_32)) && (size % 4) == 0)
	{
		sfmt_fill_array32(m_sfmt_32, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = random_32();
	}
}

void CRandom::fill_array(uint64_t* array, int32_t size) const
{
	if ((size >= sfmt_get_min_array_size64(m_sfmt_64)) && (size % 2) == 0)
	{
		sfmt_fill_array64(m_sfmt_64, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = random_64();
	}
}

void CRandom::fill_array_oc(float64_t* array, int32_t size) const
{
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_open_close(m_dsfmt, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = dsfmt_genrand_open_close(m_dsfmt);
	}
}

void CRandom::fill_array_co(float64_t* array, int32_t size) const
{
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_close_open(m_dsfmt, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = dsfmt_genrand_close_open(m_dsfmt);
	}
}

void CRandom::fill_array_oo(float64_t* array, int32_t size) const
{
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_open_open(m_dsfmt, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = dsfmt_genrand_open_open(m_dsfmt);
	}
}

void CRandom::fill_array_c1o2(float64_t* array, int32_t size) const
{
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_close1_open2(m_dsfmt, array, size);
	}
	else
	{
		for (int32_t i=0; i < size; i++)
			array[i] = dsfmt_genrand_close1_open2(m_dsfmt);
	}
}

float64_t CRandom::random_close() const
{
	return sfmt_genrand_real1(m_sfmt_32);
}

float64_t CRandom::random_open() const
{
	return dsfmt_genrand_open_open(m_dsfmt);
}

float64_t CRandom::random_half_open() const
{
	return dsfmt_genrand_close_open(m_dsfmt);
}
