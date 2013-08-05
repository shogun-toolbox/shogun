/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/distributions/classical/ProbabilityDistribution.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

CProbabilityDistribution::CProbabilityDistribution() : CSGObject()
{
	init();
}

CProbabilityDistribution::CProbabilityDistribution(int32_t dimension) :
		CSGObject()
{
	init();

	REQUIRE(dimension>0, "Dimension of Distribution must be positive\n",
			dimension);

	m_dimension=dimension;
}


CProbabilityDistribution::~CProbabilityDistribution()
{

}

SGMatrix<float64_t> CProbabilityDistribution::sample(int32_t num_samples,
		SGMatrix<float64_t> pre_samples) const
{
	SG_ERROR("Not implemented in sub-class\n");
	return SGMatrix<float64_t>();
}

SGVector<float64_t> CProbabilityDistribution::sample() const
{
	SGMatrix<float64_t> s=sample(1);
	SGVector<float64_t> result(m_dimension);
	memcpy(result.vector, s.matrix, m_dimension*sizeof(float64_t));
	return result;
}

SGVector<float64_t> CProbabilityDistribution::log_pdf_multiple(
		SGMatrix<float64_t> samples) const
{
	SG_ERROR("Not implemented in sub-class\n");
	return SGVector<float64_t>();
}

float64_t CProbabilityDistribution::log_pdf(SGVector<float64_t> single_sample) const
{
	REQUIRE(single_sample.vlen==m_dimension, "Sample dimension (%d) does not "
			"match dimension of distribution (%d)\n", single_sample.vlen,
			m_dimension);

	SGMatrix<float64_t> s(m_dimension, 1);
	memcpy(s.matrix, single_sample.vector, m_dimension*sizeof(float64_t));
	return log_pdf_multiple(s)[0];
}

void CProbabilityDistribution::init()
{
	m_dimension=0;

	SG_ADD(&m_dimension, "dimension", "Dimension of distribution.",
			MS_NOT_AVAILABLE);
}
