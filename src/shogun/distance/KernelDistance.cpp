/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/distance/KernelDistance.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CKernelDistance::CKernelDistance() : CDistance()
{
	init();
}

CKernelDistance::CKernelDistance(float64_t w, CKernel* k)
: CDistance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	SG_REF(kernel);
}

CKernelDistance::CKernelDistance(
	CFeatures *l, CFeatures *r, float64_t w , CKernel* k)
: CDistance()
{
	init();

	kernel=k;
	width=w;
	ASSERT(kernel)
	SG_REF(kernel);

	init(l, r);
}

CKernelDistance::~CKernelDistance()
{
	// important to have the cleanup of CDistance first, it calls get_name which
	// uses the distance
	cleanup();
	SG_UNREF(kernel);
}

bool CKernelDistance::init(CFeatures* l, CFeatures* r)
{
	ASSERT(kernel)
	kernel->init(l,r);
	return CDistance::init(l,r);
}

float64_t CKernelDistance::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=kernel->kernel(idx_a, idx_b);
	return exp(-result/width);
}

void CKernelDistance::init()
{
	kernel = NULL;
	width = 0.0;

	m_parameters->add(&width, "width", "Width of RBF Kernel");
	watch_param("width", &width, AnyParameterProperties("Width of RBF Kernel"));

	m_parameters->add((CSGObject**) &kernel, "kernel",
					  "Kernel.");
	watch_param(
	    "kernel", (CSGObject**)&kernel, AnyParameterProperties("kernel."));
}
