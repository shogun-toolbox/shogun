/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/KernelLocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/arpack.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/common.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>
#include <shogun/base/Parallel.h>

#ifdef HAVE_PTHREAD
#include <pthread.h>
#endif

using namespace shogun;

CKernelLocalTangentSpaceAlignment::CKernelLocalTangentSpaceAlignment() :
		CKernelLocallyLinearEmbedding()
{
}

CKernelLocalTangentSpaceAlignment::CKernelLocalTangentSpaceAlignment(CKernel* kernel) :
		CKernelLocallyLinearEmbedding(kernel)
{
}

CKernelLocalTangentSpaceAlignment::~CKernelLocalTangentSpaceAlignment()
{
}

const char* CKernelLocalTangentSpaceAlignment::get_name() const
{
	return "KernelLocalTangentSpaceAlignment";
};

#endif /* HAVE_LAPACK */
