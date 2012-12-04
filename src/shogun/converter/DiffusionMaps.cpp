/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Sergey Lisitsyn
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/converter/DiffusionMaps.h>
#include <shogun/converter/EmbeddingConverter.h>
#include <shogun/lib/config.h>
#ifdef HAVE_LAPACK
#include <shogun/mathematics/arpack.h>
#include <shogun/mathematics/lapack.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/Kernel.h>
#include <shogun/lib/Signal.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>

using namespace shogun;

CDiffusionMaps::CDiffusionMaps() :
		CEmbeddingConverter()
{
	m_t = 10;
	set_kernel(new CGaussianKernel(10,1.0));

	init();
}

void CDiffusionMaps::init()
{
	SG_ADD(&m_t, "t", "number of steps", MS_AVAILABLE);
}

CDiffusionMaps::~CDiffusionMaps()
{
}

void CDiffusionMaps::set_t(int32_t t)
{
	m_t = t;
}

int32_t CDiffusionMaps::get_t() const
{
	return m_t;
}

const char* CDiffusionMaps::get_name() const
{
	return "DiffusionMaps";
};

CFeatures* CDiffusionMaps::apply(CFeatures* features)
{
	ASSERT(features);
	// shorthand for simplefeatures
	SG_REF(features);
	// compute distance matrix
	ASSERT(m_kernel);
	m_kernel->init(features,features);
	CDenseFeatures<float64_t>* embedding = embed_kernel(m_kernel);
	m_kernel->cleanup();
	SG_UNREF(features);
	return (CFeatures*)embedding;
}

CDenseFeatures<float64_t>* CDiffusionMaps::embed_kernel(CKernel* kernel)
{
}
#endif /* HAVE_LAPACK */
