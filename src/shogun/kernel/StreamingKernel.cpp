/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 */

#include <shogun/kernel/StreamingKernel.h>
#include <shogun/features/streaming/StreamingFeatures.h>

using namespace shogun;

CStreamingKernel::CStreamingKernel()
{
	init();
}

CStreamingKernel::CStreamingKernel(CStreamingFeatures* streaming_lhs,
		CStreamingFeatures* streaming_rhs, CKernel* baseline_kernel):
		CKernel()
{
	CStreamingKernel::init();

	m_baseline_kernel=baseline_kernel;
	SG_REF(baseline_kernel);

	m_streaming_lhs=streaming_lhs;
	SG_REF(m_streaming_lhs);

	m_streaming_rhs=streaming_rhs;
	SG_REF(m_streaming_rhs);
}

CStreamingKernel::~CStreamingKernel()
{
	SG_UNREF(m_baseline_kernel);
	SG_UNREF(m_streaming_lhs);
	SG_UNREF(m_streaming_rhs);
}

float64_t CStreamingKernel::compute(int32_t idx_a, int32_t idx_b)
{
	/* NOTE: indices are ignored */

	/* create feature objects from streaming features (one each) */
	CFeatures* l=m_streaming_lhs->get_streamed_features(1);
	CFeatures* r=m_streaming_rhs->get_streamed_features(1);
	SG_REF(l);
	SG_REF(r);

	/* evaluate kernel on single features */
	m_baseline_kernel->init(l, r);
	float64_t result=m_baseline_kernel->compute(0, 0);

	/* clean up and return */
	SG_UNREF(l);
	SG_UNREF(r);
	return result;
}

SGMatrix<float64_t> CStreamingKernel::get_kernel_matrix()
{
	/* create feature objects from streaming features */
	CFeatures* l=m_streaming_lhs->get_streamed_features(m_block_size);
	CFeatures* r=m_streaming_rhs->get_streamed_features(m_block_size);
	SG_REF(l);
	SG_REF(r);

	/* compute kernel matrix on streamed features */
	m_baseline_kernel->init(l, r);
	SGMatrix<float64_t> K=m_baseline_kernel->get_kernel_matrix();

	/* clean up and return */
	SG_UNREF(l);
	SG_UNREF(r);
	return K;
}

void CStreamingKernel::init()
{
	m_baseline_kernel=NULL;
	m_streaming_lhs=NULL;
	m_streaming_rhs=NULL;
	m_block_size=1;

	SG_WARNING("TODO init method register parameters\n");
}

bool CStreamingKernel::init(CFeatures* l, CFeatures* r)
{
	CStreamingFeatures* streaming_l=dynamic_cast<CStreamingFeatures*>(l);
	CStreamingFeatures* streaming_r=dynamic_cast<CStreamingFeatures*>(r);

	REQUIRE(streaming_l, "%s::init(): LHS features must be streaming "
			"features!\n", get_name());
	REQUIRE(streaming_r, "%s::init(): RHS features must be streaming "
			"features!\n", get_name());

	return CKernel::init(l, r);
}
