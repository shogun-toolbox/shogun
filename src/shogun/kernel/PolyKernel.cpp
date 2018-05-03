/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Kyle McQuisten
 */

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/kernel/PolyKernel.h>
#include <shogun/kernel/normalizer/SqrtDiagKernelNormalizer.h>
#include <shogun/features/DotFeatures.h>

using namespace shogun;

CPolyKernel::CPolyKernel() : CDotKernel(0)
{
	init();

}

CPolyKernel::CPolyKernel(int32_t size, int32_t d, bool i) : CDotKernel(size)
{
	init();
	degree = d;
	inhomogene = i;
}

CPolyKernel::CPolyKernel(
    CDotFeatures* l, CDotFeatures* r, int32_t d, bool i, int32_t size)
    : CDotKernel(size)
{
	init();

	degree = d;
	inhomogene = i;

	init(l,r);
}

CPolyKernel::~CPolyKernel()
{
	cleanup();
}

bool CPolyKernel::init(CFeatures* l, CFeatures* r)
{
	CDotKernel::init(l,r);
	return init_normalizer();
}

void CPolyKernel::cleanup()
{
	CKernel::cleanup();
}

float64_t CPolyKernel::compute(int32_t idx_a, int32_t idx_b)
{
	float64_t result=CDotKernel::compute(idx_a, idx_b);

	if (inhomogene)
		result+=1;

	return CMath::pow(result, degree);
}

void CPolyKernel::init()
{
	degree = 0;
	inhomogene = false;

	set_normalizer(new CSqrtDiagKernelNormalizer());
	SG_ADD(&degree, "degree", "Degree of polynomial kernel", MS_AVAILABLE);
	SG_ADD(&inhomogene, "inhomogene", "If kernel is inhomogeneous.",
			MS_NOT_AVAILABLE);
}

