/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/Isomap.h>
#ifdef HAVE_LAPACK
#include <shogun/lib/common.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/Parallel.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/CoverTree.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CIsomap::CIsomap() : CMultidimensionalScaling()
{
	m_k = 3;

	init();
}

void CIsomap::init()
{
	SG_ADD(&m_k, "k", "number of neighbors", MS_AVAILABLE);
}

CIsomap::~CIsomap()
{
}

void CIsomap::set_k(int32_t k)
{
	ASSERT(k>0);
	m_k = k;
}

int32_t CIsomap::get_k() const
{
	return m_k;
}

const char* CIsomap::get_name() const
{
	return "Isomap";
}

#endif /* HAVE_LAPACK */
