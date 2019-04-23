/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Weijie Lin, Roman Votyakov, Soeren Sonnenburg
 */

#include <shogun/distributions/classical/ProbabilityDistribution.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>

using namespace shogun;

ProbabilityDistribution::ProbabilityDistribution() : SGObject()
{
	init();
}

ProbabilityDistribution::ProbabilityDistribution(int32_t dimension) :
		SGObject()
{
	init();

	REQUIRE(dimension>0, "Dimension of Distribution must be positive\n",
			dimension);

	m_dimension=dimension;
}


ProbabilityDistribution::~ProbabilityDistribution()
{

}

SGMatrix<float64_t> ProbabilityDistribution::sample(int32_t num_samples,
		SGMatrix<float64_t> pre_samples) const
{
	SG_ERROR("Not implemented in sub-class\n");
	return SGMatrix<float64_t>();
}

SGVector<float64_t> ProbabilityDistribution::sample() const
{
	SGMatrix<float64_t> s=sample(1);
	SGVector<float64_t> result(m_dimension);
	sg_memcpy(result.vector, s.matrix, m_dimension*sizeof(float64_t));
	return result;
}

SGVector<float64_t> ProbabilityDistribution::log_pdf_multiple(
		SGMatrix<float64_t> samples) const
{
	SG_ERROR("Not implemented in sub-class\n");
	return SGVector<float64_t>();
}

float64_t ProbabilityDistribution::log_pdf(SGVector<float64_t> sample_vec) const
{
	REQUIRE(sample_vec.vlen==m_dimension, "Sample dimension (%d) does not "
			"match dimension of distribution (%d)\n", sample_vec.vlen,
			m_dimension);

	SGMatrix<float64_t> s(m_dimension, 1);
	sg_memcpy(s.matrix, sample_vec.vector, m_dimension*sizeof(float64_t));
	return log_pdf_multiple(s)[0];
}

void ProbabilityDistribution::init()
{
	m_dimension=0;

	SG_ADD(&m_dimension, "dimension", "Dimension of distribution.");
}
