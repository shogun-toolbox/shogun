/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/LocallyLinearEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/base/DynArray.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLocallyLinearEmbedding::CLocallyLinearEmbedding() :
		CEmbeddingConverter()
{
	m_k = 10;
	m_max_k = 60;
	m_auto_k = false;
	m_nullspace_shift = -1e-9;
	m_reconstruction_shift = 1e-3;
	init();
}

void CLocallyLinearEmbedding::init()
{
	SG_ADD(&m_auto_k, "auto_k",
      "whether k should be determined automatically in range", MS_AVAILABLE);
	SG_ADD(&m_k, "k", "number of neighbors", MS_AVAILABLE);
	SG_ADD(&m_max_k, "max_k",
      "maximum number of neighbors used to compute optimal one", MS_AVAILABLE);
	m_parameters->add(&m_nullspace_shift, "nullspace_shift",
      "nullspace finding regularization shift");
	SG_ADD(&m_reconstruction_shift, "reconstruction_shift",
      "shift used to regularize reconstruction step", MS_NOT_AVAILABLE);
}


CLocallyLinearEmbedding::~CLocallyLinearEmbedding()
{
}

void CLocallyLinearEmbedding::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CLocallyLinearEmbedding::get_k() const
{
	return m_k;
}

void CLocallyLinearEmbedding::set_max_k(int32_t max_k)
{
	ASSERT(max_k>=m_k);
	m_max_k = max_k;
}

int32_t CLocallyLinearEmbedding::get_max_k() const
{
	return m_max_k;
}

void CLocallyLinearEmbedding::set_auto_k(bool auto_k)
{
	m_auto_k = auto_k;
}

bool CLocallyLinearEmbedding::get_auto_k() const
{
	return m_auto_k;
}

void CLocallyLinearEmbedding::set_nullspace_shift(float64_t nullspace_shift)
{
	m_nullspace_shift = nullspace_shift;
}

float64_t CLocallyLinearEmbedding::get_nullspace_shift() const
{
	return m_nullspace_shift;
}

void CLocallyLinearEmbedding::set_reconstruction_shift(float64_t reconstruction_shift)
{
	m_reconstruction_shift = reconstruction_shift;
}

float64_t CLocallyLinearEmbedding::get_reconstruction_shift() const
{
	return m_reconstruction_shift;
}

const char* CLocallyLinearEmbedding::get_name() const
{
	return "LocallyLinearEmbedding";
}

CFeatures* CLocallyLinearEmbedding::apply(CFeatures* features)
{
}

#endif /* HAVE_LAPACK */
