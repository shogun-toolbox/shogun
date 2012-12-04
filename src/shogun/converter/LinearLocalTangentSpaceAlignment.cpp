/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Sergey Lisitsyn
 */

#include <shogun/converter/LinearLocalTangentSpaceAlignment.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/Time.h>
#include <shogun/distance/Distance.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CLinearLocalTangentSpaceAlignment::CLinearLocalTangentSpaceAlignment() :
		CLocalTangentSpaceAlignment()
{
}

CLinearLocalTangentSpaceAlignment::~CLinearLocalTangentSpaceAlignment()
{
}

const char* CLinearLocalTangentSpaceAlignment::get_name() const
{
	return "LinearLocalTangentSpaceAlignment";
}

#endif /* HAVE_LAPACK */
