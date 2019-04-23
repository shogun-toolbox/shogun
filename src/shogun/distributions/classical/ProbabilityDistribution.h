/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Roman Votyakov, 
 *          Fernando Iglesias, Yuyu Zhang
 */

#ifndef PROBABILITYDISTRIBUTION_H
#define PROBABILITYDISTRIBUTION_H

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/SGMatrix.h>

namespace shogun
{

template <class T> class SGVector;

/** @brief A base class for representing n-dimensional probability distribution
 * over the real numbers (64bit) for which various statistics can be computed
 * and which can be sampled.
 */
class ProbabilityDistribution: public SGObject
{
public:
	/** Default constructor */
	ProbabilityDistribution();

	/** Constructur that sets the distribution's dimension */
	ProbabilityDistribution(int32_t dimension);

	/** Destructor */
	virtual ~ProbabilityDistribution();

	/** Samples from the distribution multiple times
	 *
	 * @param num_samples number of samples to generate
	 * @param pre_samples a matrix of pre-samples that might be used for
	 * sampling. For example, a matrix with standard normal samples for the
	 * GaussianDistribution. For reproducible results. Ignored by default.
	 * @return matrix with samples (column vectors)
	 */
	virtual SGMatrix<float64_t> sample(int32_t num_samples,
			SGMatrix<float64_t> pre_samples=SGMatrix<float64_t>()) const;

	/** Samples from the distribution once. Wrapper method. No pre-sample
	 * passing is possible with this method.
	 *
	 * @return vector with single sample
	 */
	virtual SGVector<float64_t> sample() const;

	/** Computes the log-pdf for all provided samples
	 *
	 * @param samples samples to compute log-pdf of (column vectors)
	 * @return vector with log-pdfs of given samples
	 */
	virtual SGVector<float64_t> log_pdf_multiple(SGMatrix<float64_t> samples) const;

	/** Computes the log-pdf for a single provided sample. Wrapper method which
	 * calls log_pdf_multiple
	 *
	 * @param sample_vec sample_vec to compute log-pdf for
	 * @return log-pdf of the given sample
	 */
	virtual float64_t log_pdf(SGVector<float64_t> sample_vec) const;

	/** @return name of the SGSerializable */
	virtual const char* get_name() const=0;

private:

	/** Initialses and registers parameters */
	void init();

protected:
	/** Dimension of the distribution */
	int32_t m_dimension;
};

}

#endif // PROBABILITYDISTRIBUTION_H
