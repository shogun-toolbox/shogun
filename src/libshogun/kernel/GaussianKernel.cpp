/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/common.h"
#include "kernel/GaussianKernel.h"
#include "features/Features.h"
#include "features/SimpleFeatures.h"
#include "lib/io.h"

#ifdef HAVE_BOOST_SERIALIZATION
#include <boost/serialization/export.hpp>
BOOST_CLASS_EXPORT(shogun::CGaussianKernel);
#endif //HAVE_BOOST_SERIALIZATION

using namespace shogun;

CGaussianKernel::CGaussianKernel()
: CSimpleKernel<float64_t>(), width(1)
{
}


CGaussianKernel::CGaussianKernel(int32_t size, float64_t w)
: CSimpleKernel<float64_t>(size), width(w)
{
}

CGaussianKernel::CGaussianKernel(
	CSimpleFeatures<float64_t>* l, CSimpleFeatures<float64_t>* r, float64_t w, int32_t size)
: CSimpleKernel<float64_t>(size), width(w)
{
	init(l,r);
}

CGaussianKernel::~CGaussianKernel()
{
}

bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	CSimpleKernel<float64_t>::init(l, r);
	return init_normalizer();
}

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
	int32_t alen, blen;
	bool afree, bfree;

	float64_t* avec=((CSimpleFeatures<float64_t>*) lhs)->get_feature_vector(idx_a, alen, afree);
	float64_t* bvec=((CSimpleFeatures<float64_t>*) rhs)->get_feature_vector(idx_b, blen, bfree);
	ASSERT(alen==blen);

	float64_t result=0;
	for (int32_t i=0; i<alen; i++)
		result+=CMath::sq(avec[i]-bvec[i]);

	result=exp(-result/width);

	((CSimpleFeatures<float64_t>*) lhs)->free_feature_vector(avec, idx_a, afree);
	((CSimpleFeatures<float64_t>*) rhs)->free_feature_vector(bvec, idx_b, bfree);

	return result;
}
