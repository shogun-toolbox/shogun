/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/converter/NeighborhoodPreservingEmbedding.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/lib/FibonacciHeap.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CNeighborhoodPreservingEmbedding::CNeighborhoodPreservingEmbedding() :
		CLocallyLinearEmbedding()
{
}

CNeighborhoodPreservingEmbedding::~CNeighborhoodPreservingEmbedding()
{
}

const char* CNeighborhoodPreservingEmbedding::get_name() const
{
	return "NeighborhoodPreservingEmbedding";
}

#endif /* HAVE_LAPACK */
